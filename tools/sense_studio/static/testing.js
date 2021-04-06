
function addTerminalMessage(message) {
    let terminal = document.getElementById('testTerminal');
    terminal.insertAdjacentHTML('beforeend', `<div class='monospace-font'><b>${message}</b></div>`);
    terminal.scrollTop = terminal.scrollHeight;
}

function streamVideo(message) {
    let frame = document.getElementById('frame');
    frame.src = message.image;
}

async function getPathInputs(pathLabelId, pathInputId) {
    let pathLabel = document.getElementById(pathLabelId);
    let pathInput = document.getElementById(pathInputId);
    let project = document.getElementById('project');

    let name = project.value;
    let path = pathInput.value;

    let directoriesResponse = await browseDirectory(path, name);
    let disabled = false;

    // Check that project path is filled and exists
    if (path === '') {
        setFormWarning(pathLabel, pathInput, '');
        disabled = true;
    } else if (!directoriesResponse.path_exists) {
        setFormWarning(pathLabel, pathInput, 'This path does not exist');
        disabled = true;
    } else {
        setFormWarning(pathLabel, pathInput, '');
    }
}

async function startTesting(url){
    let classifier = document.getElementById('classifier').value;
    let inputVideoPath = document.getElementById('inputVideoPath').value;
    let outputVideoName = document.getElementById('outputVideoName').value;
    let path = document.getElementById('path').value;
    let outputFolder = document.getElementById('outputFolder').value;
    let title = document.getElementById('title').value;
    let buttonTest = document.getElementById('btnTest');
    let buttonCancelTest = document.getElementById('btnCancelTest');
    let video_stream = document.getElementById('videoStream');

    data = {
        classifier: classifier,
        inputVideoPath: inputVideoPath,
        outputVideoName: outputVideoName,
        path: path,
        outputFolder: outputFolder,
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

    socket.on('testing_images', function(message) {
        streamVideo(message);
    });

    socket.on('success', function(message) {
        if (message.status === 'Complete') {
            video_stream.classList.add('uk-hidden');
            addTerminalMessage('Stopping Inference...');

            socket.disconnect();
            console.log('Socket Disconnected');

            buttonTest.disabled = false;
            buttonCancelTest.disabled = true;
        }
    });

    socket.on('testing_logs', function(message) {
        video_stream.classList.add('uk-hidden');
        addTerminalMessage(message.log);
    });

    buttonTest.disabled = true;
    buttonCancelTest.disabled = false;

    video_stream.classList.remove('uk-hidden');
    addTerminalMessage('Starting Inference...');
}

async function cancelTesting(url){
    await asyncRequest(url);

    document.getElementById('btnTest').disabled = false;
    document.getElementById('btnCancelTest').disabled = true;
    document.getElementById('videoStream').classList.add('uk-hidden');
}
