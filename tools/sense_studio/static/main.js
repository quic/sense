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
    let pathSearchInputs = document.getElementsByClassName('path-search');
    for (input of pathSearchInputs) {
        const currentInput = input;
        new autoComplete({
            selector: input,
            minChars: 1,
            cache: false,
            source: function(term, response) {
                let directoriesResponse = browseDirectory(term, '');
                response(directoriesResponse.subdirs);
            },
            onSelect: function(event, term, item) {
                currentInput.oninput();
            }
        });
    }

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

    $('.ui .dropdown').dropdown();

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
    let nameInput = document.getElementById('newProjectName');
    let nameLabel = document.getElementById('newProjectNameLabel');
    let pathInput = document.getElementById('newProjectPath');
    let pathLabel = document.getElementById('newProjectPathLabel');
    let fullPathDiv = document.getElementById('fullPath');
    let createProjectButton = document.getElementById('createProject');

    let name = nameInput.value;
    let path = pathInput.value;

    let directoriesResponse = browseDirectory(path, name);
    fullPathDiv.innerHTML = directoriesResponse.full_project_path;

    let disabled = false;

    // Check that project name is filled, unique and not yet present in directory
    if (name === '') {
        setFormWarning(nameLabel, nameInput, '');
        disabled = true;
    } else if (!directoriesResponse.project_name_unique) {
        setFormWarning(nameLabel, nameInput, 'This project name is already used');
        disabled = true;
    } else if (directoriesResponse.full_path_exists) {
        setFormWarning(nameLabel, nameInput, 'A directory with this name already exists in the chosen location');
        disabled = true;
    } else {
        setFormWarning(nameLabel, nameInput, '');
    }

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

    createProjectButton.disabled = disabled;
}


function editImportProject() {
    let pathInput = document.getElementById('importProjectPath');
    let pathLabel = document.getElementById('importProjectPathLabel');
    let importProjectButton = document.getElementById('importProject');

    let path = pathInput.value;

    let directoriesResponse = browseDirectory(path, '');

    let disabled = false;

    // Check that project path is filled, unique and exists
    if (path === '') {
        setFormWarning(pathLabel, pathInput, '');
        disabled = true;
    } else if (!directoriesResponse.path_unique) {
        setFormWarning(pathLabel, pathInput, 'Another project is already registered in this location');
        disabled = true;
    } else if (!directoriesResponse.path_exists) {
        setFormWarning(pathLabel, pathInput, 'This path does not exist');
        disabled = true;
    } else {
        setFormWarning(pathLabel, pathInput, '');
    }

    importProjectButton.disabled = disabled;
}


function editUpdateProject(projectIdx) {
    let pathInput = document.getElementById(`updateProjectPath${projectIdx}`);
    let pathLabel = document.getElementById(`updateProjectLabel${projectIdx}`);
    let updateProjectButton = document.getElementById(`updateProject${projectIdx}`);

    let path = pathInput.value;

    let directoriesResponse = browseDirectory(path, '');

    let disabled = false;

    // Check that project path is filled, unique and exists
    if (path === '') {
        setFormWarning(pathLabel, pathInput, '');
        disabled = true;
    } else if (!directoriesResponse.path_unique) {
        setFormWarning(pathLabel, pathInput, 'Another project is already registered in this location');
        disabled = true;
    } else if (!directoriesResponse.path_exists) {
        setFormWarning(pathLabel, pathInput, 'This path does not exist');
        disabled = true;
    } else {
        setFormWarning(pathLabel, pathInput, '');
    }

    updateProjectButton.disabled = disabled;
}


$.fn.form.settings.rules.uniqueClassName = function (className) {
    let projectName = $('#projectName').val();
    let config = getProjectConfig(projectName);
    return !(className in config.classes)
}


function asyncRequest(url, data, callback) {
    return new Promise(function (resolve, reject) {
        let xhttp = new XMLHttpRequest();

        xhttp.onload = function () {
            response = JSON.parse(xhttp.responseText);
            resolve(response);
        };

        if (data) {
            xhttp.open('POST', url, true);
            xhttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');
            xhttp.send(JSON.stringify(data));
        } else {
            xhttp.open('GET', url, true);
            xhttp.send();
        }
    });
}


function syncRequest(url, data) {
    let xhttp = new XMLHttpRequest();

    if (data) {
        xhttp.open('POST', url, false);
        xhttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');
        xhttp.send(JSON.stringify(data));
    } else {
        xhttp.open('GET', url, false);
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


async function prepareAnnotations(element, projectName) {
    loading(element, 'Preparing Annotations');

    await asyncRequest('/annotation/prepare-annotation', {projectName: projectName});

    loadingDone(element, 'Annotations Prepared');
}


function loading(element, message, url) {
    let icon = element.children[0];
    let text = element.children[1];

    icon.removeAttribute('uk-icon');
    icon.setAttribute('uk-spinner', 'ratio: 0.6');
    text.innerHTML = message;
    element.disabled = true;

    if (url) {
        window.location = url;
    }
}


function loadingDone(element, message) {
    let icon = element.children[0];
    let text = element.children[1];

    icon.removeAttribute('uk-spinner');
    icon.classList.remove('uk-spinner');
    icon.setAttribute('uk-icon', 'icon: check');
    text.innerHTML = message;
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


function toggleAssistedTagging(path, split, label) {
    response = syncRequest('/toggle-project-setting',
                           {path: path, setting: 'assisted_tagging', split: split, label: label});

    // Reload page to update predictions
    window.location.reload();
}
