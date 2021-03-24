// Suppress Enter key on text input fields for submitting forms, so that for example search fields don't submit
// the form on selection of an entry
window.addEventListener(
    'keydown',
    function(e) {
        if ((e.keyIdentifier=='U+000A' || e.keyIdentifier=='Enter' || e.keyCode==13)
            && e.target.nodeName=='INPUT' && e.target.type=='text')
        {
            e.preventDefault();
            return false;
        }
    },
    true
);


$(document).ready(function () {
    new autoComplete({
        selector: '.path-search',
        minChars: 1,
        cache: false,
        source: function(term, response) {
            let directoriesResponse = browseDirectory(term, '');
            response(directoriesResponse.subdirs);
        },
        onSelect: function(event, term, item) {
            let inputs = document.getElementsByClassName('path-search');

            for (input of inputs) {
                if (input.value === term) {
                    input.oninput();
                }
            }
        }
    });

    $('.class-card').form({
        fields: {
            className: {
                rules: [
                    {
                        type   : 'empty',
                        prompt : 'Please enter a class name'
                    },
                    {
                        type   : 'uniqueClassName',
                        prompt : 'The chosen class name already exists'
                    }
                ]
            }
        }
    });

});


function setFormWarning(label, input, text) {
    label.innerHTML = text;

    if (text === '') {
        input.classList.remove('uk-form-danger');
    } else {
        input.classList.add('uk-form-danger');
    }
}


function editNewProject() {
    let newProjectNameInput = document.getElementById('newProjectName');
    let newProjectNameLabel = document.getElementById('newProjectNameLabel');
    let newProjectPathInput = document.getElementById('newProjectPath');
    let newProjectPathLabel = document.getElementById('newProjectPathLabel');
    let createProjectButton = document.getElementById('createProject');
    let fullPathDiv = document.getElementById('fullPath');

    let newProjectName = newProjectNameInput.value;
    let newProjectPath = newProjectPathInput.value;

    let directoriesResponse = browseDirectory(newProjectPath, newProjectName);
    fullPathDiv.innerHTML = directoriesResponse.full_project_path;

    let disabled = false;

    // Check that project name is filled, unique and not yet present in directory
    if (newProjectName === '') {
        setFormWarning(newProjectNameLabel, newProjectNameInput, '');
        disabled = true;
    } else if (!directoriesResponse.project_name_unique) {
        setFormWarning(newProjectNameLabel, newProjectNameInput, 'This project name is already used');
        disabled = true;
    } else if (directoriesResponse.full_path_exists) {
        setFormWarning(newProjectNameLabel, newProjectNameInput,
                       'A directory with this name already exists in the chosen location');
        disabled = true;
    } else {
        setFormWarning(newProjectNameLabel, newProjectNameInput, '');
    }

    // Check that project path is filled and exists
    if (newProjectPath === '') {
        setFormWarning(newProjectPathLabel, newProjectPathInput, '');
        disabled = true;
    } else if (!directoriesResponse.path_exists) {
        setFormWarning(newProjectPathLabel, newProjectPathInput, 'This path does not exist');
        disabled = true;
    } else {
        setFormWarning(newProjectPathLabel, newProjectPathInput, '');
    }

    createProjectButton.disabled = disabled;
}


function editImportProject() {
    let importProjectPathInput = document.getElementById('importProjectPath');
    let importProjectPathLabel = document.getElementById('importProjectPathLabel');
    let importProjectButton = document.getElementById('importProject');

    let importProjectPath = importProjectPathInput.value;

    let directoriesResponse = browseDirectory(importProjectPath, '');

    let disabled = false;

    // Check that project path is filled, unique and exists
    if (importProjectPath === '') {
        setFormWarning(importProjectPathLabel, importProjectPathInput, '');
        disabled = true;
    } else if (!directoriesResponse.path_unique) {
        setFormWarning(importProjectPathLabel, importProjectPathInput,
                       'Another project is already registered in this location');
        disabled = true;
    } else if (!directoriesResponse.path_exists) {
        setFormWarning(importProjectPathLabel, importProjectPathInput, 'This path does not exist');
        disabled = true;
    } else {
        setFormWarning(importProjectPathLabel, importProjectPathInput, '');
    }

    importProjectButton.disabled = disabled;
}


$.fn.form.settings.rules.uniqueClassName = function (className) {
    let projectName = $('#projectName').val();
    let config = getProjectConfig(projectName);
    return !(className in config.classes)
}


function syncRequest(url, data) {
    let xhttp = new XMLHttpRequest();

    xhttp.open('POST', url, false);
    xhttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');

    if (data) {
        xhttp.send(JSON.stringify(data));
    } else {
        xhttp.send();
    }

    return JSON.parse(xhttp.responseText);
}


function browseDirectory(path, projectName) {
    return syncRequest('/browse-directory', {path: path, project: projectName});
}


function getProjectConfig(projectName) {
    return syncRequest('/project-config', {name: projectName});
}


function loading(element) {
    element.classList.add('loading');
    element.classList.add('disabled');
}


// TODO: Tag colors still need to be adapted
const buttonClasses = [
    'uk-button-primary',
    'uk-button-secondary',
    'uk-button-danger',
]


function assignTag(frameIdx, selectedTagIdx) {
    let tagInput = document.getElementById(`${frameIdx}_tag`);
    tagInput.value = selectedTagIdx;

    for (const tagIdx of [0, 1, 2]) {
        let button = document.getElementById(`${frameIdx}_tag${tagIdx}`);

        if (tagIdx == selectedTagIdx) {
            button.classList.add(buttonClasses[tagIdx]);
        } else {
            button.classList.remove(buttonClasses[tagIdx]);
        }
    }
}


function initTagButtons(annotations) {
    for (let frameIdx = 0; frameIdx < annotations.length; frameIdx++) {
        assignTag(frameIdx, annotations[frameIdx]);
    }
}


function editClass(index, shouldEdit) {
    let classShow = document.getElementById(`classShow${index}`);
    let classEdit = document.getElementById(`classEdit${index}`);

    if (shouldEdit) {
        classShow.classList.add('uk-hidden');
        classEdit.classList.remove('uk-hidden');
    } else {
        classShow.classList.remove('uk-hidden');
        classEdit.classList.add('uk-hidden');
    }
}


function toggleGPU(path) {
    response = syncRequest('/toggle-project-setting', {path: path, setting: 'use_gpu'});

    let gpuInput = document.getElementById('gpuInput');
    gpuInput.checked = response.setting_status;
}


function toggleMakeProjectTemporal(path) {
    response = syncRequest('/toggle-project-setting', {path: path, setting: 'temporal'});

    let makeProjectTemporal = document.getElementById('makeProjectTemporal');
    let temporalElements = document.getElementsByClassName('temporal');

    makeProjectTemporal.checked = response.setting_status;

    // Show/hide all temporal-related elements
    for (element of temporalElements) {
        if (response.setting_status) {
            element.classList.remove('uk-hidden');
        } else {
            element.classList.add('uk-hidden');
        }
    }
}


function toggleShowPredictions(path) {
    response = syncRequest('/toggle-project-setting', {path: path, setting: 'show_logreg'});

    let logregInput = document.getElementById('logregInput');
    let logregElements = document.getElementsByClassName('logreg-predictions');

    logregInput.checked = response.setting_status;

    // Show/hide all LogReg prediction-labels
    for (element of logregElements) {
        if (response.setting_status) {
            element.classList.remove('uk-hidden');
        } else {
            element.classList.add('uk-hidden');
        }
    }
}
