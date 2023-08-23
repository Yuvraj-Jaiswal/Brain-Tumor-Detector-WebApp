from flask import Flask , request , jsonify , make_response , render_template , redirect , url_for , send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from flask_migrate import Migrate
import matplotlib.pyplot as plt
from functools import wraps
from ultralytics import YOLO
from time import sleep
import numpy as np
import requests
import glob
import PIL
import base64
import datetime
import cvzone
import cv2
import jwt
import os
import re
import io
import json


app = Flask(__name__)
app.config['SECRET_KEY'] = "yuvraj_flask2"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.getcwd(), 'database.db')
app.config['UPLOAD_FOLDER'] = "static/images"
db = SQLAlchemy(app)
migrate = Migrate(app, db)
model = YOLO(model='Brain-tumor-model-600.pt')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(20),nullable=False)
    def __repr__(self):
        return f'username :  {self.username} , email : {self.email}'

with app.app_context():
    db.create_all()

def add_user(username:str,email:str,password:str):
    try:
        new_object = User(username=username, email=email , password=password)
        db.session.add(new_object)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(e)

def delete_user(user_id : int):
    try:
        user = User.query.get(user_id)
        db.session.delete(user)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(e)

def checkMail(email:str):
    count = email.count(".")
    double_dot = email.count("..")
    re_pattern = r"^[A-Za-z0-9_!#$%&'*+\/=?`{|}~^.-]+@[A-Za-z0-9.-]+$"
    if re.match(re_pattern, email) and email[-1] != "." and email[-2] != "." and len(email) > 0 and double_dot == 0 and 1 <= count <= 2:
        return True
    else: return False

def get_list_data(data):
    formatted_data = [{'id': user.id, 'username': user.username , 'email' : user.email , 'password' : user.password} for user in data]
    return formatted_data


@app.route("/" , methods=['GET', 'POST'])
def login():
    if request.method == "POST" and "loginSb" in request.form:
        username = request.form['username']
        password = request.form['password']
        user_data = {
            "username" : username,
            "password" : password
        }
        response = requests.post("http://127.0.0.1:5000/login" ,json=user_data)
        print(response)
        if response.status_code == 200:
            token = response.json()['access_token']
            return redirect(url_for('home', token=token))
        else:
            return render_template("login.html" , messageL=response.json()['message'] , dataL=request.form)


    elif request.method == "POST" and "registerSb" in request.form:
        data = request.form
        user_data = {
            'username' : data['username'],
            'email' : data['email'],
            'password' : data['password']
        }
        response = requests.post("http://127.0.0.1:5000/register" , json=user_data)
        return render_template("login.html" , messageR=response.json()['message'] , dataR=data)
    return render_template("login.html")

def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    total_size_mb = total_size / (1024 * 1024)
    return total_size_mb


def delete_files(path, max_size_mb):
    folder_size_mb = get_folder_size(path)
    if folder_size_mb > max_size_mb:
        print(f"Folder size ({folder_size_mb} MB) exceeds {max_size_mb} MB. Deleting files...")
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                os.remove(filepath)
        print("Files deleted.")


@app.route("/home" , methods=["GET","POST"])
def home():
    try:
        token = request.args.get("token")
        data = jwt.decode(token, app.config["SECRET_KEY"], algorithms="HS256")
        if data['exp_time'] < datetime.datetime.now().isoformat():
            return jsonify({"message": "Token is expired generate new token"}), 403
    except Exception as e:
        print(e)
        return jsonify({"message": "Invalid Token Error"}), 403

    if request.method == "POST":
        label_list = ['glioma', 'meningioma', 'pituitary']
        delete_files("static/images",80)

        if 'imgFile' in request.files:
            image = request.files['imgFile']

            if image.filename == '':
                return jsonify({'class': 'No file selected'}), 404

            if allowed_file(image.filename):
                filename = image.filename
                byte_stream = io.BytesIO(image.read())
                image = PIL.Image.open(byte_stream)
                image = np.array(image)
                img = cv2.resize(image , (640, 640))
                results = model(img)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        cls = int(box.cls[0])
                        label = label_list[cls]

                        conf = box.conf[0]
                        conf = float(conf * 100)
                        conf = round(conf)
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), 10, 2, 1)
                        cv2.rectangle(img, (x1, y1), (x1 + 110, y1 - 30), color=(255, 255, 255), thickness=-1)
                        cv2.putText(img, f"{str(label).capitalize()} {conf}%", (x1 + 3, y1 - 12),
                                    cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, thickness=1, color=(255, 0, 255))
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detections_' + filename)
                cv2.imwrite(save_path, img)
                return jsonify({'class' : "image detection success" , 'path' : save_path})
            else:
                return jsonify({'class': 'Invalid file format. Only JPG, JPEG, and PNG files are allowed.'}), 404
    return render_template("home.html" )


@app.route('/users' , methods=['GET'])
def get_user_api():
    data = User.query.all()
    formatted_data = get_list_data(data)
    return jsonify(formatted_data)


@app.route('/users/<int:user_id>' , methods=['DELETE' , 'GET'])
def delete_user_api(user_id):
    user = User.query.get(user_id)
    if user is None:
        return jsonify({'message': f'User with id - {user_id} does not exists'}), 400

    if request.method == 'GET':
        json_data = {'id': user.id, 'username': user.username , 'email' : user.email , 'password' : user.password}
        return jsonify(json_data), 200

    elif request.method == 'DELETE':
        delete_user(user_id)
        return jsonify({'message': f'User with user id : {user_id} deleted successfully'}) , 200

def tokenizer(f):
    @wraps(f)
    def decorated(*args , **kwargs):
        token = request.headers.get("token")

        if not token:
            return jsonify({"message" : "Token is required"}) , 403

        try:
            data = jwt.decode(token , app.config["SECRET_KEY"] ,algorithms="HS256")
            if data['exp_time'] < datetime.datetime.now().isoformat():
                return jsonify({"message": "Token is expired generate new token"}), 403

        except Exception as e:
            print(e)
            return jsonify({"message" : "Token is invalid"}) , 403
        return f(data , *args , **kwargs)

    return decorated


@app.route('/register', methods=['POST'])
def register_api():
    username = request.json.get('username')
    email = request.json.get('email')
    password = request.json.get('password')

    if not username or not password or not email:
        return jsonify({'message': 'missing required fields - username , email , password'}), 400

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'message': 'Username already exists'}), 400

    if checkMail(email) is False:
        return jsonify({'message': 'Enter valid email'}), 400

    existing_email = User.query.filter_by(email=email).first()
    if existing_email:
        return jsonify({'message': 'Email already exists'}), 400

    if len(password) < 8:
        return jsonify({'message': 'Password must be at least 8 characters long'}), 400

    add_user(username , email , generate_password_hash(password))
    return jsonify({'message': 'User registered successfully , Now Login'}), 201



@app.route('/login', methods=['POST'])
def login_api():
    username = request.json.get('username')
    password = request.json.get('password')

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'message': 'Invalid username or password'}), 401

    expiration_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    expiration_time_str = expiration_time.isoformat()

    token = jwt.encode({
        "id" : user.id ,
        "exp_time" : expiration_time_str } , app.config["SECRET_KEY"] , algorithm="HS256"
            )

    return jsonify({'access_token': token}), 200


@app.route('/protected', methods=['GET'])
@tokenizer
def protected(data):
    auth_user = User.query.get(int(data.get("id")))
    return jsonify({'message': 'Protected route accessed successfully' , 'user' : auth_user.username}) , 200


def allowed_file(filename):
    extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


@app.route('/upload', methods=['POST'])
@tokenizer
def upload_file(data):
    if 'file' not in request.files:
        return jsonify({'message': 'No file uploaded.'}) , 404

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected.'}) , 404

    label_list = ['giloma', 'pitutary', 'migioma']

    if file and allowed_file(file.filename):
        img = PIL.Image.open(file)
        print(img)
        img = img.resize((640, 640),resample=PIL.Image.NEAREST)
        img = np.array(img)
        results = model(img)

        classes = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                label = label_list[cls]
                classes.append(label)

                conf = box.conf[0]
                conf = float(conf * 100)
                conf = round(conf)

                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), 10, 2, 1)

                cv2.rectangle(img, (x1, y1), (x1 + 110, y1 - 30), color=(255, 255, 255), thickness=-1)
                cv2.putText(img, f"{str(label).capitalize()} {conf}%", (x1 + 3, y1 - 12),
                            cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, thickness=1, color=(255, 0, 255))

        processed_image = Image.fromarray(img)
        image_file = io.BytesIO()
        processed_image.save(image_file, 'JPEG')
        image_file.seek(0)
        return send_file(image_file, mimetype='image/jpeg')

    else:
        return jsonify({'message' : 'Invalid file format. Only JPG, JPEG, and PNG files are allowed.'}) , 400


if __name__ == '__main__':
    app.run(debug=True)