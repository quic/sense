
function addTerminalMessage(message) {
    let terminal = document.getElementById('terminal');
    terminal.insertAdjacentHTML('beforeend', `<div class='monospace-font'><b>${message}</b></div>`);
    terminal.scrollTop = terminal.scrollHeight;
}


async function startTraining(url) {

    let project = document.getElementById('project').value;
    let path = document.getElementById('path').value;
    let layersToFinetune = document.getElementById('layersToFinetune').value;
    let outputFolder = document.getElementById('outputFolder').value;
    let modelName = document.getElementById('modelName').value;
    let epochs = document.getElementById('epochs').value;

    let terminal = document.getElementById('terminal');
    let buttonTrain = document.getElementById('btnTrain');
    let buttonCancelTrain = document.getElementById('btnCancelTrain');
    let confusionMatrix = document.getElementById('confusionMatrix');

    data = {
        path: path,
        layersToFinetune: layersToFinetune,
        outputFolder: outputFolder,
        modelName: modelName,
        epochs: epochs,
    };
    await asyncRequest(url, data);

    let socket = io.connect('/connect-training-logs');
    socket.on('connect', function() {
        console.log('Socket Connected');
        socket.emit('connect_training_logs',
                    {status: 'Socket Connected', project: project, outputFolder: outputFolder});
    });

    socket.on('training_logs', function(message) {
        addTerminalMessage(message.log);
    });

    socket.on('success', function(message) {
        if (message.status === 'Complete') {
            socket.disconnect();
            console.log('Socket Disconnected');

            buttonTrain.disabled = false;
            buttonCancelTrain.disabled = true;
            confusionMatrix.innerHTML = `<img src=${message.img_path} alt='Confusion matrix' />`;
        }
    });

    socket.on('failed', function(message) {
        if (message.status === 'Failed') {
            socket.disconnect();
            console.log('Socket Disconnected');

            buttonTrain.disabled = false;
            buttonCancelTrain.disabled = true;
        }
    });

    buttonTrain.disabled = true;
    buttonCancelTrain.disabled = false;
    terminal.innerHTML = '';
    confusionMatrix.innerHTML = '';

    addTerminalMessage('Training started...');
}


async function cancelTraining(url) {
    await asyncRequest(url);

    document.getElementById('btnTrain').disabled = false;
    document.getElementById('btnCancelTrain').disabled = true;

    addTerminalMessage('Training cancelled.');
}
