
function increase(inputID) {
    let element = document.getElementById(inputID);
    let value = parseInt(element.value);

    if (isNaN(value)) {
        value = 1;
    } else {
        value++;
    }

    element.value = value;

    enableSetDefaultsButton();
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

    enableSetDefaultsButton();
}


function enableSetDefaultsButton() {
    let setDefaultButton = document.getElementById('setDefaultButton');
    setDefaultButton.classList.remove('disabled');
    setDefaultButton.innerHTML = "Save as defaults"
}


function recordVideo(url) {
    // Check if ffmpeg is installed
    response = syncRequest('/video-recording/ffmpeg-check')
    if (!response.ffmpeg_installed) {
        displayOverlay('Please make sure ffmpeg is installed!', 'error');
        return;
    }

    document.getElementById('recordVideoButton').classList.add('disabled');
    navigator.mediaDevices.getUserMedia({ audio: false, video: true })
        .then(stream => setupRecording(stream, url));
}


function displayOverlay(text, mode) {
    let overlay = document.getElementById('textOverlay');
    let container = document.getElementById('videoContainer');

    // Update overlay text
    overlay.innerHTML = text;

    // Update overlay visibility
    if (text !== '') {
        overlay.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
    }

    // Update overlay color
    if (mode === 'saved') {
        overlay.classList.add('green');
        overlay.classList.remove('red');
    } else if (mode === 'error') {
        overlay.classList.remove('green');
        overlay.classList.add('red');
    } else {
        overlay.classList.remove('green');
        overlay.classList.remove('red');
    }

    // Update container color
    if (mode === 'recording') {
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

    let countdownDuration = parseInt(document.getElementById('countdown').value)
    let recordingDuration = parseInt(document.getElementById('duration').value)

    // Show countdown
    for (const seconds of Array(countdownDuration).keys()) {
        const countdown = countdownDuration - seconds;
        setTimeout(displayOverlay, seconds * 1000, `Get Ready: ${countdown}`, 'countdown');
    }

    // Start recording
    setTimeout(startRecording, countdownDuration * 1000, stream, recordingDuration, url);
};


function startRecording(stream, recordingDuration, url) {
    let mediaRecorder = new MediaRecorder(stream, {mimeType: 'video/webm; codecs=vp8'});
    mediaRecorder.ondataavailable = function (event) {
        if (event.data.size > 0) {
            saveVideo(event.data, url);
        }
    };
    mediaRecorder.start();

    // Show countdown
    for (const seconds of Array(recordingDuration).keys()) {
        const countdown = recordingDuration - seconds;
        setTimeout(displayOverlay, seconds * 1000, countdown, 'recording');
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
    document.getElementById('recordVideoButton').classList.remove('disabled');
}


function saveVideo(chunk, url) {
    let blob = new Blob([chunk], {type: 'video/webm'});
    const formData = new FormData();
    formData.append('video', blob);
    fetch(url, {
        method: 'POST',
        body: formData,
    }).then(res => {
        if (res.ok) {
            displayOverlay('Video Saved', 'saved');
        } else {
            displayOverlay(`Error: ${res.status}`, 'error');
        }
    }).catch(err => {
        displayOverlay(err, 'error');;
    });
}
