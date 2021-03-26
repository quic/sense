
$(document).ready(function () {
    $('.path-search').search({
        apiSettings: {
            response: function (e) {
                let path = this.children[1].value;
                let response = browseDirectory(path);

                let results = [];
                for (const subdir of response.subdirs) {
                    results.push({title: subdir})
                }

                return {results: results}
            }
        },
        showNoResults: false,
        cache: false
    });

    $('.update-project-card').form({
        fields: {
            path: {
                rules: [
                    {
                        type   : 'regExp',
                        value  : /^\/.*/,
                        prompt : 'Please enter an absolute path (starting with a "/")'
                    },
                    {
                        type   : 'notExistingPath',
                        prompt : 'The chosen directory doesn\'t exist'
                    },
                    {
                        type   : 'uniquePath',
                        prompt : 'Another project is already initialized in this location'
                    }
                ]
            }
        }
    });

    $('#newProjectCard').form({
        fields: {
            projectName: {
                rules: [
                    {
                        type   : 'empty',
                        prompt : 'Please enter a project name'
                    },
                    {
                        type   : 'uniqueProjectName',
                        prompt : 'The chosen project name already exists'
                    }
                ]
            },
            path: {
                rules: [
                    {
                        type   : 'regExp',
                        value  : /^\/.*/,
                        prompt : 'Please enter an absolute path (starting with a "/")'
                    },
                    {
                        type   : 'uniquePath',
                        prompt : 'Another project is already initialized in this location'
                    }
                ]
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

    $('.hasclickpopup').popup({
        inline: true,
        on: 'click',
        position: 'bottom right',
    });

    $('.hashoverpopup').popup();

    $('.display-hidden').hide();

    $('.message .close').on('click', function() {
        $(this).closest('.message').transition('fade');
    });

});


$.fn.form.settings.rules.uniqueProjectName = function (projectName) {
    let projects = getProjects();
    let projectNames = Object.keys(projects);
    return !projectNames.includes(projectName);
}


$.fn.form.settings.rules.existingPath = function (projectPath) {
    let response = browseDirectory(projectPath);
    return !response.path_exists;
}


$.fn.form.settings.rules.notExistingPath = function (projectPath) {
    if (projectPath) {
        let response = browseDirectory(projectPath);
        return response.path_exists;
    } else {
        return true;
    }
}


$.fn.form.settings.rules.uniquePath = function (projectPath) {
    let projects = getProjects();
    for (project of Object.values(projects)) {
        if (project.path === projectPath) {
            return false;
        }
    }
    return true;
}


$.fn.form.settings.rules.uniqueClassName = function (className) {
    let projectName = $('#projectName').val();
    let config = getProjectConfig(projectName);
    return !(className in config.classes)
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


function getProjects() {
    return syncRequest('/projects-list');
}


function browseDirectory(path) {
    return syncRequest('/browse-directory', {path: path});
}


function getProjectConfig(projectName) {
    return syncRequest('/project-config', {name: projectName});
}


function loading(element) {
    element.classList.add('loading');
    element.classList.add('disabled');
}


let tagColors = [
    'grey',
    'blue',
    'green'
];


function assignTag(frameIdx, selectedTagIdx) {
    let tagInput = document.getElementById(`${frameIdx}_tag`);
    tagInput.value = selectedTagIdx;

    for (const tagIdx of [0, 1, 2]) {
        let button = document.getElementById(`${frameIdx}_tag${tagIdx}`);

        if (tagIdx == selectedTagIdx) {
            button.classList.add(tagColors[tagIdx]);
        } else {
            button.classList.remove(tagColors[tagIdx]);
        }
    }
}


function initTagButtons(annotations) {
    for (let frameIdx = 0; frameIdx < annotations.length; frameIdx++) {
        assignTag(frameIdx, annotations[frameIdx]);
    }
}


function editClass(index, shouldEdit) {
    let classShow = $(`#classShow${index}`);
    let classEdit = $(`#classEdit${index}`);

    if (shouldEdit) {
        classShow.hide();
        classEdit.show();
    } else {
        classShow.show();
        classEdit.hide();
    }
}


function toggleGPU(path) {
    let gpuInput = document.getElementById('gpuInput');
    response = syncRequest('/toggle-project-setting', {path: path, setting: 'use_gpu'});

    if (response.setting_status) {
        gpuInput.setAttribute('checked', 'checked');
    } else {
        gpuInput.removeAttribute('checked');
    }
}


function toggleMakeProjectTemporal(path) {
    let makeProjectTemporal = document.getElementById('makeProjectTemporal');
    response = syncRequest('/toggle-project-setting', {path: path, setting: 'temporal'});

    // Show/hide all temporal-related elements
    if (response.setting_status) {
        makeProjectTemporal.setAttribute('checked', 'checked');
        $('.temporal').show();
    } else {
        makeProjectTemporal.removeAttribute('checked');
        $('.temporal').hide();
    }
}


function toggleAssistedTagging(path, split, label) {
    let logregInput = document.getElementById('logregInput');
    response = syncRequest('/toggle-project-setting',
                           {path: path, setting: 'assisted_tagging', split: split, label: label});

    window.location.reload();
}


function setTimerDefault(path) {
    let setDefaultButton = document.getElementById('setDefaultButton');
    let countdownDuration = parseInt(document.getElementById('countdown').value);
    let recordingDuration = parseInt(document.getElementById('duration').value);

    setDefaultButton.classList.add('disabled');
    setDefaultButton.innerHTML = "Saved";

    response = syncRequest('/set-timer-default', {path: path, countdown: countdownDuration, recording: recordingDuration});
}
