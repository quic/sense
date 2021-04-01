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

function startTesting(url){
    let classifier = document.getElementById('classifier').value;
    let inputVideoPath = document.getElementById('inputVideoPath').value;
    let outputVideoName = document.getElementById('outputVideoName').value;
    let path = document.getElementById('path').value;
    let outputFolder = document.getElementById('outputFolder').value;
    let title = document.getElementById('title').value;
    let buttonTest = document.getElementById('btnTest');
    let buttonCancelTest = document.getElementById('btnCancelTest');

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

    asyncRequest(url, data);
}

async function cancelTesting(url){
    await asyncRequest(url);

    document.getElementById('btnTest').disabled = false;
    document.getElementById('btnCancelTest').disabled = true;
}
