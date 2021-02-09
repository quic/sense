
function increase(inputID) {
    let element = document.getElementById(inputID);
    let value = parseInt(element.value);

    if (isNaN(value)) {
        value = 1;
    } else {
        value++;
    }

    element.value = value;
}


function decrease(inputID, minValue) {
    let element = document.getElementById(inputID);
    let value = parseInt(element.value);

    if (isNaN(value)) {
        value = 1;
    } else if (value > minValue) {
        value--;
    }

    element.value = value;
}


function recordVideo(url) {
    navigator.mediaDevices.getUserMedia({ audio: false, video: true })
        .then(stream => setupRecording(stream, url));
}


function displayOverlay(text, recording) {
    overlay = document.getElementById('textOverlay');
    container = document.getElementById('videoContainer');

    // Update overlay
    overlay.innerHTML = text;
    if (text !== '') {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }

    // Update container color
    if (recording) {
        container.classList.add('red');
        container.classList.add('inverted');
    } else {
        container.classList.remove('red');
        container.classList.remove('inverted');
    }
}


function setupRecording(stream, url) {
    let player = document.getElementById('player');
    player.srcObject = stream;

    countdownDuration = parseInt(document.getElementById('countdown').value)
    recordingDuration = parseInt(document.getElementById('duration').value)

    // Show countdown
    for (const seconds of Array(countdownDuration).keys()) {
        const countdown = countdownDuration - seconds;
        setTimeout(displayOverlay, seconds * 1000, 'Get Ready: ' + countdown, false);
    }

    // Start recording
    setTimeout(startRecording, countdownDuration * 1000, stream, recordingDuration, url);
};


function startRecording(stream, recordingDuration, url) {
    mediaRecorder = new MediaRecorder(stream, {mimeType: 'video/webm; codecs=vp8'});
    mediaRecorder.ondataavailable = function (event) {
        if (event.data.size > 0) {
            saveVideo(event.data, url);
        }
    };
    mediaRecorder.start();

    // Show countdown
    for (const seconds of Array(recordingDuration).keys()) {
        const countdown = recordingDuration - seconds;
        setTimeout(displayOverlay, seconds * 1000, countdown, true);
    }

    // Stop recording
    setTimeout(stopRecording, recordingDuration * 1000, mediaRecorder);
}


function stopRecording(mediaRecorder) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(function(track) {
        track.stop();
    });

    displayOverlay('', false);
}


function saveVideo(chunk, url) {
    let blob = new Blob([chunk], {type: "video/webm"});
    const formData = new FormData();
    formData.append('video', blob);
    fetch(url, {
        method: 'POST',
        body: formData,
    }).then(res => {
        if(res.ok) alert('Video saved');
    }).catch(err => {
        alert(err);
    });
}
