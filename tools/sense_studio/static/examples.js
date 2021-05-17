function toggleIrrelevantFields() {
    let example = document.getElementById('example');
    let metToCalorieConverters = document.getElementById('metToCalorieConverters');

    // Script names containing MET converters (e.g. weight, height, age, gender)
    // If add new example with any converter from above, add that script name here in below list (without .py)
    let examplesWithMETConverters = ["run_calorie_estimation", "run_fitness_tracker"];

    if (examplesWithMETConverters.includes(example.value)) {
        metToCalorieConverters.classList.remove('uk-hidden');
    } else {
        metToCalorieConverters.classList.add('uk-hidden');
    }
}


function addTerminalMessage(message) {
    let terminal = document.getElementById('demoTerminal');
    terminal.insertAdjacentHTML('beforeend', `<div class='monospace-font'><b>${message}</b></div>`);
    terminal.scrollTop = terminal.scrollHeight;
}

function streamVideo(message) {
    let frame = document.getElementById('frame');
    frame.src = message.image;
}

async function startDemo(url) {
    let example = document.getElementById('example');
    let weight = document.getElementById('weight').value;
    let age = document.getElementById('age').value;
    let height = document.getElementById('height').value;
    let gender = document.getElementById('gender').value;
    let modelName = document.getElementById('modelName').value;
    let webcamInput = document.getElementsByName('inputSource')[0];
    let saveVideo = document.getElementById('saveVideo');
    let inputVideoPath = document.getElementById('inputVideoPath');
    let inputVideoPathValue = (webcamInput.checked) ? '' : inputVideoPath.value;
    let outputVideoName = document.getElementById('outputVideoName');
    let outputVideoNameValue = (saveVideo.checked) ? outputVideoName.value : '';
    let path = document.getElementById('path').value;
    let title = document.getElementById('title').value;
    let buttonRunDemo = document.getElementById('btnRunDemo');
    let buttonCancelDemo = document.getElementById('btnCancelDemo');
    let frame = document.getElementById('frame');

    data = {
        example: example.options[example.selectedIndex].text,
        modelName: modelName,
        inputVideoPath: inputVideoPathValue,
        outputVideoName: outputVideoNameValue,
        height: height,
        weight: weight,
        age: age,
        gender: gender,
        path: path,
        title: title,
    };

    buttonRunDemo.disabled = true;
    buttonCancelDemo.disabled = false;

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

            buttonCancelDemo.disabled = true;

            // Enable Test button again depending on input fields
            checkInputFields();
        }
    });

    socket.on('example_logs', function(message) {
        addTerminalMessage(message.log);
    });

    addTerminalMessage('Starting Inference...');
}

async function cancelDemo(url) {
    addTerminalMessage('Stopping Inference...');
    await asyncRequest(url);

    document.getElementById('btnCancelDemo').disabled = true;

    // Enable run example button again depending on input fields
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

    let buttonRunDemo = document.getElementById('btnRunDemo');
    let buttonCancelDemo = document.getElementById('btnCancelDemo');

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

    // Don't enable the run example button, if the model is currently running (i.e. Cancel button is enabled)
    if (!buttonCancelDemo.disabled) {
        disabled = true;
    }

    buttonRunDemo.disabled = disabled;
}


