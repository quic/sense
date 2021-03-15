
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

    xhttp.open('POST', url, false);
    xhttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');

    if (data) {
        xhttp.send(JSON.stringify(data));
    } else {
        xhttp.send();
    }

    return JSON.parse(xhttp.responseText);
}


function getProjects() {
    return syncRequest('/projects-list', null);
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
