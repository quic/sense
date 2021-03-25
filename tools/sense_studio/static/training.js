let socket = null;

function addTerminalMessage(message) {
    $("#terminal").append(`<p class='monospace-font'><b>${message}</b></p>`);
    $('#terminal').scrollTop($('#terminal')[0].scrollHeight);
}


function startTraining(url) {

    let project = $('#project').val();
    let path = $('#path').val();
    let layersToFinetune = $('#layersToFinetune').val();
    let outputFolder = $('#outputFolder').val();
    let modelName = $('#modelName').val();
    let epochs = $('#epochs').val();

    socket = io.connect('/connect-training-logs');
    socket.on('connect', function() {
        console.log('Socket Connected');
        socket.emit('connect_training_logs',
                    {status: 'Socket Connected', project: project, outputFolder: outputFolder});
    });

    socket.on('status', function(message) {
        console.log(message.status);
    });

    socket.on('training_logs', function(message) {
        addTerminalMessage(message.log);
    });

    socket.on('success', function(message) {
        if (message.status === 'Complete') {
            socket.disconnect();
            console.log('Socket Disconnected');

            $('#btnTrain').removeClass('disabled');
            $('#btnCancelTrain').addClass('disabled');
            $('#confusionMatrix').append(`<img src=${message.img_path} alt='Confusion matrix' />`);
            $('#confusionMatrix').show();
        }
    });

    socket.on('failed', function(message) {
        if (message.status === 'Failed') {
            socket.disconnect();
            console.log('Socket Disconnected');

            $('#btnTrain').removeClass('disabled');
            $('#btnCancelTrain').addClass('disabled');
        }
    });

    $('#btnTrain').addClass('disabled');
    $('#btnCancelTrain').removeClass('disabled');
    $('#terminal').html('');
    $('#confusionMatrix').html('');

    addTerminalMessage('Training started...');

    data = {
        path: path,
        layersToFinetune: layersToFinetune,
        outputFolder: outputFolder,
        modelName: modelName,
        epochs: epochs,
    };
    syncRequest(url, data);
}


function cancelTraining(url) {
    syncRequest(url);

    socket.disconnect();
    console.log('Socket Disconnected');

    $('#btnTrain').removeClass('disabled');
    $('#btnCancelTrain').addClass('disabled');

    addTerminalMessage('Training cancelled');
}
