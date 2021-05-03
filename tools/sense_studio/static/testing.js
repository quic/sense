
function addTerminalMessage(message) {
    let terminal = document.getElementById('testTerminal');
    terminal.insertAdjacentHTML('beforeend', `<div class='monospace-font'><b>${message}</b></div>`);
    terminal.scrollTop = terminal.scrollHeight;
}

function streamVideo(message) {
    let frame = document.getElementById('frame');
    frame.src = message.image;
}

async function startTesting(url) {
    let classifier = document.getElementById('classifier').value;
    let webcamInput = document.getElementsByName('inputSource')[0];
    let saveVideo = document.getElementById('saveVideo');
    let inputVideoPath = document.getElementById('inputVideoPath');
    let inputVideoPathValue = (webcamInput.checked) ? '' : inputVideoPath.value;
    let outputVideoName = document.getElementById('outputVideoName');
    let outputVideoNameValue = (saveVideo.checked) ? outputVideoName.value : '';
    let path = document.getElementById('path').value;
    let title = document.getElementById('title').value;
    let buttonTest = document.getElementById('btnTest');
    let buttonCancelTest = document.getElementById('btnCancelTest');
    let frame = document.getElementById('frame');

    data = {
        classifier: classifier,
        inputVideoPath: inputVideoPathValue,
        outputVideoName: outputVideoNameValue,
        path: path,
        title: title,
    };

    buttonTest.disabled = true;
    buttonCancelTest.disabled = false;

    await asyncRequest(url, data);

    let socket = io.connect('/stream-video');
    socket.on('connect', function() {
        console.log('Socket Connected');
        socket.emit('stream_video', {status: 'Socket Connected'});
    });

    socket.on('stream_frame', function(message) {
        streamVideo(message);
    });

    socket.on('success', function(message) {
        if (message.status === 'Complete') {
            frame.removeAttribute('src');
            socket.disconnect();
            console.log('Socket Disconnected');

            buttonCancelTest.disabled = true;

            // Enable Test button again depending on input fields
            checkInputFields();
        }
    });

    socket.on('testing_logs', function(message) {
        addTerminalMessage(message.log);
    });

    addTerminalMessage('Starting Inference...');
}

async function cancelTesting(url) {
    addTerminalMessage('Stopping Inference...');
    await asyncRequest(url);

    document.getElementById('btnCancelTest').disabled = true;

    // Enable Test button again depending on input fields
    checkInputFields();
}

function toggleInputVideoField() {
    let webcamInput = document.getElementsByName('inputSource')[0];
    let inputVideoDiv = document.getElementById('inputVideoDiv');

    if (webcamInput.checked) {
        inputVideoDiv.classList.add('uk-hidden');
    } else {
        inputVideoDiv.classList.remove('uk-hidden');
    }

    checkInputFields();
}

function toggleOutputVideoField() {
    let saveVideo = document.getElementById('saveVideo');
    let outputVideoDiv = document.getElementById('outputVideoDiv');

    if (saveVideo.checked) {
        outputVideoDiv.classList.remove('uk-hidden');
    } else {
        outputVideoDiv.classList.add('uk-hidden');
    }

    checkInputFields();
}

async function checkInputFields() {
    let webcamInput = document.getElementsByName('inputSource')[0];
    let inputVideoPathLabel = document.getElementById('inputVideoPathLabel');
    let inputVideoPath = document.getElementById('inputVideoPath');
    let inputVideoPathValue = inputVideoPath.value;

    let saveVideo = document.getElementById('saveVideo');
    let outputVideoNameLabel = document.getElementById('outputVideoNameLabel');
    let outputVideoName = document.getElementById('outputVideoName');
    let outputVideoNameValue = outputVideoName.value;

    let buttonTest = document.getElementById('btnTest');
    let buttonCancelTest = document.getElementById('btnCancelTest');

    let project = document.getElementById('project');
    let projectName = project.value;

    let directoriesResponse = await browseDirectory(inputVideoPathValue, projectName);

    let disabled = false;

    // Check that input video path is filled and exists if not streaming from webcam
    if (webcamInput.checked) {
        setFormWarning(inputVideoPathLabel, inputVideoPath, '');
    } else if (inputVideoPathValue === '') {
        setFormWarning(inputVideoPathLabel, inputVideoPath, 'Please provide an input video');
        disabled = true;
    } else if (!directoriesResponse.path_exists) {
        setFormWarning(inputVideoPathLabel, inputVideoPath, 'This path does not exist');
        disabled = true;
    } else if (!inputVideoPathValue.endsWith('.mp4')) {
        setFormWarning(inputVideoPathLabel, inputVideoPath, 'Please provide a valid .mp4 file');
        disabled = true;
    } else {
        // Correct path provided
        setFormWarning(inputVideoPathLabel, inputVideoPath, '');
    }

    // Check that output video name is provided if video should be saved
    if (!saveVideo.checked) {
        setFormWarning(outputVideoNameLabel, outputVideoName, '');
    } else if (!outputVideoNameValue) {
        setFormWarning(outputVideoNameLabel, outputVideoName, 'Please provide a video name');
        disabled = true;
    } else {
        // Name provided
        setFormWarning(outputVideoNameLabel, outputVideoName, '');
    }

    // Don't enable the Test button, if the model is currently running (i.e. Cancel button is enabled)
    if (!buttonCancelTest.disabled) {
        disabled = true;
    }

    buttonTest.disabled = disabled;
}
