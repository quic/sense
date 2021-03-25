var socket;
$(document).ready(function () {

    let project = $('#project').val();
    let outputFolder = $('#outputFolder').val();

    // TODO: Remove http://
    socket = io.connect('http://' + document.domain + ':' + location.port + '/train-model');
    socket.on('connect', function() {
        socket.emit('training_logs', {status: 'Socket Connected', project: project});
    });

    socket.on('status', function(message) {
        console.log(message.status);
    });

    socket.on('training_logs', function(message) {
        $("#terminal").children().append(`<p class='monospace-font'><b>${message.log}</b></p>`);
        $('#terminal').scrollTop($('#terminal')[0].scrollHeight);
    });

    socket.on('success', function(message) {
        if (message.status === 'Complete') {
            socket.disconnect();
            console.log('Socket Disconnected');

            $('#btnTrain').removeClass('disabled');
            $('#btnCancelTrain').addClass('disabled');
            $('#confusionMatrix').append(`<img src=${message.img_path}/${outputFolder} alt='Confusion matrix' />`);
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
});
