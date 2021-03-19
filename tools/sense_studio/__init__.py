from flask import Flask
from flask_socketio import SocketIO


socketio = SocketIO()


def init_app(debug=False):
    app = Flask(__name__)
    app.secret_key = 'd66HR8dç"f_-àgjYYic*dh'
    app.debug = debug

    from tools.sense_studio.annotation import annotation_bp
    app.register_blueprint(annotation_bp, url_prefix='/annotation')

    from tools.sense_studio.video_recording import video_recording_bp
    app.register_blueprint(video_recording_bp, url_prefix='/video-recording')

    from tools.sense_studio.training import training_bp
    app.register_blueprint(training_bp, url_prefix='/training')

    socketio.init_app(app, cors_allowed_origins="*")

    return app
