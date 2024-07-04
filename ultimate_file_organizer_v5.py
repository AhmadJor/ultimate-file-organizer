import os
import shutil
import re
import logging
import threading
import json
import sqlite3
import zipfile
import smtplib
import schedule
import time
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyPDF2 import PdfFileReader
import docx
import filetype
from textblob import TextBlob
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup  # For generating HTML reports
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# Load configuration from JSON file
def load_config():
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        raise
    except json.JSONDecodeError:
        logging.error("Error decoding JSON configuration file.")
        raise

config = load_config()

directories = config.get('directories', {})
patterns = {key: re.compile(pattern, re.IGNORECASE) for key, pattern in config.get('patterns', {}).items()}
email_config = config.get('email', {})

# Validate configuration data
def validate_config():
    if not directories:
        raise ValueError("Directories for categorization not specified in configuration.")
    if not patterns:
        raise ValueError("Patterns for categorization not specified in configuration.")
    if not email_config.get('from') or not email_config.get('to') or not email_config.get('smtp_server'):
        raise ValueError("Email configuration is incomplete.")

validate_config()

# Setup logging
logging.basicConfig(filename='file_organizer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup database
conn = sqlite3.connect('file_organizer.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS actions
             (timestamp TEXT, action TEXT, filename TEXT, category TEXT, new_filename TEXT, recipient TEXT)''')

# Define regex patterns for identifying recipients
recipient_patterns = [
    re.compile(r'\bName:\s*([A-Za-z\s]+)\b', re.IGNORECASE),
    re.compile(r'\bStudent:\s*([A-Za-z\s]+)\b', re.IGNORECASE),
    re.compile(r'\bRecipient:\s*([A-Za-z\s]+)\b', re.IGNORECASE),
    # Add more patterns as needed
]

def extract_recipient(text):
    for pattern in recipient_patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return "Unknown"

def handle_duplicate(target_dir, filename):
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(target_dir, new_filename)):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    return new_filename

def read_file_content(file_path, file_extension):
    text = ""
    if file_extension in ['.pdf']:
        with open(file_path, 'rb') as f:
            pdf = PdfFileReader(f)
            for page_num in range(min(2, pdf.getNumPages())):  # Read first two pages
                text += pdf.getPage(page_num).extract_text() or ""
    elif file_extension in ['.docx']:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs[:10]])  # Read first 10 paragraphs
    elif file_extension in ['.txt', '.py', '.html', '.css', '.js']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    return text

def categorize_file_by_content(file_path, filename):
    file_extension = os.path.splitext(filename)[1].lower()
    category = 'Others'
    recipient = "Unknown"

    try:
        text = read_file_content(file_path, file_extension)

        if text:
            if re.search(patterns['Reports'], text):
                category = 'Reports'
            elif re.search(patterns['Assignments'], text):
                category = 'Assignments'
            
            # Extract recipient information
            recipient = extract_recipient(text)

        # Additional NLP-based categorization
        text_blob = TextBlob(text)
        if 'analysis' in text_blob.lower():
            category = 'Reports'
        elif 'assignment' in text_blob.lower() or 'homework' in text_blob.lower():
            category = 'Assignments'
    except Exception as e:
        logging.error(f"Error categorizing file by content: {e}")
    
    return category, recipient

def classify_file_with_ml(model, file_path):
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        content = read_file_content(file_path, file_extension)
        if content:
            category = model.predict([content])[0]
            return category
    except Exception as e:
        logging.error(f"Error classifying file with machine learning: {e}")
    return 'Others'

def organize_file(model, file_path, filename):
    file_extension = os.path.splitext(filename)[1].lower()
    base_name = os.path.splitext(filename)[0].lower()
    recipient = "Unknown"
    
    try:
        # Check patterns for categorization
        for category, pattern in patterns.items():
            if pattern.search(base_name):
                return category, recipient

        # Check file type by content if not categorized by name
        kind = filetype.guess(file_path)
        if kind:
            mime_type = kind.mime.split('/')[0]
            if mime_type in ['image']:
                return 'Images', recipient
            elif mime_type in ['audio']:
                return 'Audio', recipient
            elif mime_type in ['video']:
                return 'Video', recipient
        
        # Check content for categorization
        content_category, recipient = categorize_file_by_content(file_path, filename)
        if content_category != 'Others':
            return content_category, recipient

        # Machine Learning classification
        ml_category = classify_file_with_ml(model, file_path)
        if ml_category:
            return ml_category, recipient
    except Exception as e:
        logging.error(f"Error organizing file {filename}: {e}")
    
    # Default to Others
    return 'Others', recipient

def compress_directory(directory):
    try:
        shutil.make_archive(directory, 'zip', directory)
        shutil.rmtree(directory)
    except Exception as e:
        logging.error(f"Error compressing directory {directory}: {e}")

def backup_files(source_dir):
    try:
        backup_dir = source_dir + "_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(source_dir, backup_dir)
        logging.info(f"Backup created at {backup_dir}")
    except Exception as e:
        logging.error(f"Error creating backup: {e}")

def restore_files(source_dir):
    try:
        backup_dir = source_dir + "_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(source_dir)
            shutil.copytree(backup_dir, source_dir)
            logging.info(f"Files restored from {backup_dir}")
        else:
            logging.warning("No backup found to restore.")
    except Exception as e:
        logging.error(f"Error restoring files: {e}")

def send_email_notification(subject, body):
    msg = MIMEMultipart()
    msg['From'] = email_config['from']
    msg['To'] = email_config['to']
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['from'], email_config['password'])
        text = msg.as_string()
        server.sendmail(email_config['from'], email_config['to'], text)
        server.quit()
        logging.info("Email notification sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

def generate_report():
    try:
        actions = c.execute("SELECT * FROM actions").fetchall()
        html = '''
        <html>
            <head>
                <title>File Organization Report</title>
            </head>
            <body>
                <h1>File Organization Report</h1>
                <table border="1">
                    <tr>
                        <th>Timestamp</th>
                        <th>Action</th>
                        <th>Filename</th>
                        <th>Category</th>
                        <th>New Filename</th>
                        <th>Recipient</th>
                    </tr>
        '''
        for action in actions:
            html += f'''
            <tr>
                <td>{action[0]}</td>
                <td>{action[1]}</td>
                <td>{action[2]}</td>
                <td>{action[3]}</td>
                <td>{action[4]}</td>
                <td>{action[5]}</td>
            </tr>
            '''
        html += '''
                </table>
            </body>
        </html>
        '''
        with open('report.html', 'w') as f:
            f.write(html)
        logging.info("Report generated as report.html")
    except Exception as e:
        logging.error(f"Failed to generate report: {e}")

class Watcher:
    def __init__(self, directory_to_watch):
        self.DIRECTORY_TO_WATCH = directory_to_watch
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except Exception as e:
            logging.error(f"Observer error: {e}")
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return None
        else:
            logging.info(f"File created: {event.src_path}")
            try:
                category, recipient = organize_file(model, event.src_path, os.path.basename(event.src_path))
                new_file_path = os.path.join(event.src_path, category, os.path.basename(event.src_path))
                if not os.path.exists(os.path.join(event.src_path, category)):
                    os.makedirs(os.path.join(event.src_path, category))
                shutil.move(event.src_path, new_file_path)
                c.execute("INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?)", (datetime.now(), 'create', os.path.basename(event.src_path), category, new_file_path, recipient))
                conn.commit()
            except Exception as e:
                logging.error(f"Error handling created file {event.src_path}: {e}")

def organize_files(source_dir):
    try:
        backup_files(source_dir)
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            logging.info(f"Processing file: {file_path}")
            if os.path.isfile(file_path):
                category, recipient = organize_file(model, file_path, filename)
                target_dir = os.path.join(source_dir, category)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                new_filename = handle_duplicate(target_dir, filename)
                new_file_path = os.path.join(target_dir, new_filename)
                shutil.move(file_path, new_file_path)
                logging.info(f"Moved '{filename}' to '{category}' as '{new_filename}' for recipient '{recipient}'")
                c.execute("INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?)", (datetime.now(), 'move', filename, category, new_filename, recipient))
                conn.commit()
        
        for dir_name in directories.keys():
            if dir_name not in ['Others']:
                compress_directory(os.path.join(source_dir, dir_name))

        send_email_notification("File Organization Completed", "The file organization process has been completed successfully.")
        generate_report()
    except Exception as e:
        logging.error(f"Error organizing files: {e}")

# GUI Interface
def run_gui():
    def choose_directory():
        directory = filedialog.askdirectory()
        if directory:
            source_dir_entry.delete(0, tk.END)
            source_dir_entry.insert(0, directory)

    def start_organization():
        source_dir = source_dir_entry.get()
        if os.path.exists(source_dir):
            try:
                organize_files(source_dir)
                messagebox.showinfo("Success", "Files have been organized. Check the log for details.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        else:
            messagebox.showwarning("Warning", "Please choose a valid directory.")

    root = tk.Tk()
    root.title("File Organizer")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    tk.Label(frame, text="Source Directory:").grid(row=0, column=0, sticky=tk.W)
    source_dir_entry = tk.Entry(frame, width=50)
    source_dir_entry.grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Browse...", command=choose_directory).grid(row=0, column=2, padx=5)

    tk.Button(frame, text="Organize Files", command=start_organization).grid(row=1, columnspan=3, pady=10)

    root.mainloop()

def scheduled_organize_files():
    schedule.every().day.at("02:00").do(lambda: organize_files(source_dir))
    while True:
        schedule.run_pending()
        time.sleep(1)

# Execute the function
if __name__ == '__main__':
    start_time = datetime.now()

    # Load the trained model
    model_path = 'file_classifier_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print(f"Model file '{model_path}' not found. Ensure you have trained the model first.")
        exit()

    run_gui()

    scheduled_organize_files()
    source_dir = "c:/Users/jorba/OneDrive/Desktop/organize"
    watcher = Watcher(source_dir)
    watcher.run()

# Close the database connection
conn.close()
