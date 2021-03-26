function getClassifierPath() {
    let classifierLabel = document.getElementById('classifierLabel');
    let classifierPath = document.getElementById('classifier');
    let project = document.getElementById('project');

    let name = project.value;
    let path = classifierPath.value;

    let directoriesResponse = browseDirectory(path, name);
    let disabled = false;

    // Check that project path is filled and exists
    if (path === '') {
        setFormWarning(classifierLabel, classifierPath, '');
        disabled = true;
    } else if (!directoriesResponse.path_exists) {
        setFormWarning(classifierLabel, classifierPath, 'This path does not exist');
        disabled = true;
    } else {
        setFormWarning(classifierLabel, classifierPath, '');
    }

}