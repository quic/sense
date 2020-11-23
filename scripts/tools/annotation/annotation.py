#!/usr/bin/env python
# -*- coding:utf-8 -*-

from flask import Flask, request, flash, redirect, url_for
from flask import render_template, send_from_directory
import os
from os.path import join
import json
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load


PROJ_DIR = '/home/amercier/code/20bn-realtimenet/annotation/0'
out_folder = '/home/amercier/code/20bn-realtimenet/annotation/0'
features_dir = join(PROJ_DIR, 'features')
tags_dir = join(PROJ_DIR, 'tags')
os.makedirs(tags_dir, exist_ok=True)

lr = None
lr_path = join(out_folder, 'lr.joblib')
if os.path.isfile(lr_path):
    lr = load(lr_path)





app = Flask(__name__)
app.secret_key = 'd66HR8dç"f_-àgjYYic*dh'

DOSSIER_UPS = join(out_folder, 'frames')
videos = os.listdir(join(PROJ_DIR, 'frames'))

videos.sort()

def extension_ok(nomfic):
    """ Renvoie True si le fichier possède une extension d'image valide. """
    return '.' in nomfic and nomfic.rsplit('.', 1)[1] in ('png', 'jpg', 'jpeg', 'gif', 'bmp')

@app.route('/annot/')
def list_annot():
    folder_id = zip(videos, list(range(len(videos))))
    return render_template('up_folder.html', folders=folder_id)

@app.route('/annot/<nom>')
def annot(nom):
    nom = int(nom)
    features = np.load(join(features_dir, videos[nom] + ".npy"))
    features = features.mean(axis=(2, 3))

    if lr is not None:
        classes = list(lr.predict(features))
    else:
        classes = [0] * len(features)
    print(classes)
    images = [img for img in glob.glob(DOSSIER_UPS + '/' + videos[nom] + '/*') if extension_ok(img)] # la liste des images dans le dossier
    nums = [int(x.split('.')[0].split('/')[-1]) for x in images]
    n_images = len(nums)
    images = [[x.replace(PROJ_DIR, ''), y] for y, x in sorted(zip(nums,images))]
    images = [[x[0], x[1], y] for x,y in zip(images, classes)]
    chunk_size = 5
    n_chunk = int(len(images)/chunk_size)
    images = np.array_split(images, n_chunk)
    images = [list(x) for x in images]
    print(n_images)
    print(images)
    return render_template('up_liste.html', images=images, num=nom, fps=16, n_images=n_images)


@app.route('/response', methods = ['POST'])
def response():
    if request.method == 'POST':
        data = request.form # a multidict containing POST data
        num = int(data['num'])
        fps = float(data['fps'])
        desc = {'file': videos[num] + ".mp4"}
        desc['fps'] = fps
        out_annotation = os.path.join(out_folder, 'tags', videos[num] + ".json")
        time_annotation = []
        for i in range(int(data['n_images'])):
            time_annotation.append(int(data[str(i)]))
        desc['time_annotation'] = time_annotation
        json.dump(desc, open(out_annotation, 'w'))

    return redirect(url_for('annot', nom=num+1))


@app.route('/train_lr', methods=['POST'])
def train_lr():
    global lr
    if request.method == 'POST':
        data = request.form # a multidict containing POST data
        num = int(data['num'])
        annotations = glob.glob(f'{tags_dir}/*.json')
        features = [x.replace('/tags/', '/features/').replace('.json',
                                                              '.npy') for x in annotations]
        X = []
        y = []
        for feature in features:
            feature = np.load(feature)
            for f in feature:
                X.append(f.mean(axis=(1, 2)))
        for annotation in annotations:
            annotation = json.load(open(annotation, 'r'))['time_annotation']
            pos1 = np.where(np.array(annotation ).astype(int) == 1)[0]
            if len(pos1) > 0:
                for p in pos1:
                    try:
                        annotation[p + 1] = 1
                    except:
                        1
            pos1 = np.where(np.array(annotation ).astype(int) == 2)[0]
            if len(pos1) > 0:
                for p in pos1:
                    try:
                        annotation[p + 1] = 2
                    except:
                        1

            for a in annotation:
                y.append(a)
        X = np.array(X)
        print(X.shape)
        y = np.array(y)
        lr = LogisticRegression(C=0.1, class_weight={0:0.5, 1:2, 2: 2})
        lr.fit(X, y)
        dump(lr, lr_path)
    return redirect(url_for('annot', nom=num))

#
# @app.route('/export_annotation', methods = ['POST'])
# def export_annotation():
#     data = request.form  # a multidict containing POST data
#     num = int(data['num'])
#     new_annotations = []
#     for label, tags in export_labels.items():
#         annotations_taged = glob.glob(os.path.join('annotations',label, 'tags','*', 'annotation.json'))
#         for an in annotations_taged:
#             an = json.load(open(an,'r'))
#             time_annotation = np.array(an.pop('time_annotation'))
#             fps = an.pop('fps')
#             ones = np.where(time_annotation == 1)[0]
#             twos = np.where(time_annotation == 2)[0]
#             ones = (ones*4 + 3)/fps
#             twos = (twos*4 + 3)/fps
#             if len(ones) + len(twos) > 0:
#                 count_tags = {}
#                 if len(ones) > 0:
#                     count_tags[tags[0]] = list(ones)
#                 if len(twos) > 0:
#                     count_tags[tags[1]] = list(twos)
#                 an['counting_tag'] = count_tags
#                 new_annotations.append(an)
#     json.dump(new_annotations, open(os.path.join('annotations', 'export_annotations.json'), 'w'))
#     return redirect(url_for('annot', nom=num))

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(PROJ_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

