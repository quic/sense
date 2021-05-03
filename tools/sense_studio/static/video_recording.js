
function enableSetDefaultsButton() {
    let setDefaultButton = document.getElementById('setDefaultButton');
    setDefaultButton.disabled = false;
    setDefaultButton.innerHTML = "Save as defaults"
}


function setTimerDefault(path) {
    let setDefaultButton = document.getElementById('setDefaultButton');
    let countdownDuration = parseInt(document.getElementById('countdown').value);
    let recordingDuration = parseInt(document.getElementById('duration').value);

    setDefaultButton.disabled = true;
    setDefaultButton.innerHTML = "Saved";

    asyncRequest('/set-timer-default', {path: path, countdown: countdownDuration, recording: recordingDuration});
}


async function recordVideo(url) {
    // Check if ffmpeg is installed
    let response = await asyncRequest('/video-recording/ffmpeg-check')
    if (!response.ffmpeg_installed) {
        displayOverlay('Please make sure ffmpeg is installed!', 'error');
        return;
    }

    document.getElementById('recordVideoButton').disabled = true;
    navigator.mediaDevices.getUserMedia({ audio: false, video: true })
        .then(stream => setupRecording(stream, url));
}


function displayOverlay(text, mode) {
    let overlay = document.getElementById('textOverlay');

    // Update overlay text
    overlay.innerHTML = text;

    // Make overlay visible
    overlay.classList.remove('uk-hidden');

    // Update overlay color
    if (mode === 'recording') {
        overlay.classList.remove('uk-text-success');
        overlay.classList.add('uk-text-danger');
    } else if (mode === 'saved') {
        overlay.classList.add('uk-text-success');
        overlay.classList.remove('uk-text-danger');
    } else if (mode === 'error') {
        overlay.classList.remove('uk-text-success');
        overlay.classList.add('uk-text-danger');
    } else {
        overlay.classList.remove('uk-text-success');
        overlay.classList.remove('uk-text-danger');
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
        setTimeout(displayOverlay, seconds * 1000, `Recording: ${countdown}`, 'recording');
    }

    // Stop recording
    setTimeout(stopRecording, recordingDuration * 1000, mediaRecorder);
}


function stopRecording(mediaRecorder) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(function(track) {
        track.stop();
    });

    displayOverlay('Saving', 'saving');
    document.getElementById('recordVideoButton').disabled = false;
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
