
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
    response = syncRequest('/toggle-project-setting', {path, 'use_gpu'});

    if (response.status) {
        gpuInput.setAttribute('checked', 'checked');
    } else {
        gpuInput.removeAttribute('checked');
    }
}


function toggleMakeProjectTemporal(path, project) {
    let makeProjectTemporal = document.getElementById('makeProjectTemporal');
    response = syncRequest('/toggle-project-setting', {path, 'temporal'});

    // Get all tags related divs
    let classTagsDiv = document.getElementsByClassName('classTags');
    let editClassTagsDiv = document.getElementsByClassName('editClassTags');
    let addClassTagsDiv = document.getElementsByClassName('addClassTags');
    let annotateButtons = document.getElementsByClassName('annotate');
    let navbarAnnotateDiv = document.getElementById('navbarAnnotate');
    let annotatedTextSpan = document.getElementsByClassName('annotatedText');
    let displayStyle = 'none';
    let visibilityStyle = 'hidden';

    if (response.status) {
        makeProjectTemporal.setAttribute('checked', 'checked');
        displayStyle = 'block';
        visibilityStyle = 'visible';

    } else {
        makeProjectTemporal.removeAttribute('checked');
    }

    // Show/Hide annotate buttons
     for (let i=0; i < annotateButtons.length; i++){
        annotateButtons[i].style.visibility = visibilityStyle;
        annotateButtons[i].style.display = displayStyle;

        annotatedTextSpan[i].style.visibility = visibilityStyle;
        annotatedTextSpan[i].style.display = 'inline';
    }

    // Show/Hide tags on project details page
    for (let i=0; i < classTagsDiv.length; i++){
        classTagsDiv[i].style.display = displayStyle;
        editClassTagsDiv[i].style.display = displayStyle;
    }
    addClassTagsDiv[0].style.display = displayStyle;

    // Show/Hide annotation button on navigation bar
    if (response.status){
        navbarAnnotateDiv.style.display = 'flex';
    } else{
        navbarAnnotateDiv.style.display = 'none';
    }
}
