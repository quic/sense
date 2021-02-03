
$(document).ready(function () {
    $('#pathSearch').search({
        apiSettings: {
            response: function (e) {
                let path = document.getElementById('path').value;
                let response = checkProjectDirectory(path);

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

    $('#newProjectForm').form({
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
                        value  : /\/.*/,
                        prompt : 'Please enter an absolute path (starting with a "/")'
                    },
                    {
                        type   : 'existingPath',
                        prompt : 'The chosen directory already exists'
                    }
                ]
            }
        }
    });

    let relocateProjectName = '';
    let nameInput = $('#name');
    if (nameInput) {
        relocateProjectName = nameInput.val();
    }
    let relocateProjectPath = '';
    let pathInput = $('#path');
    if (pathInput) {
        relocateProjectPath = pathInput.val();
    }

    $('#importProjectForm').form({
        fields: {
            name: {
                rules: [
                    {
                        type   : 'empty',
                        prompt : 'Please enter a project name'
                    },
                    {
                        type   : 'uniqueProjectName',
                        value  : relocateProjectName,
                        prompt : 'The chosen project name already exists'
                    }
                ]
            },
            path: {
                rules: [
                    {
                        type   : 'regExp',
                        value  : /\/.*/,
                        prompt : 'Please enter an absolute path (starting with a "/")'
                    },
                    {
                        type   : 'notExistingPath',
                        prompt : 'The chosen directory doesn\'t exist'
                    },
                    {
                        type   : 'uniquePath',
                        value  : relocateProjectPath,
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
});


$.fn.form.settings.rules.uniqueProjectName = function (projectName, relocatedName) {
    if (relocatedName && projectName === relocatedName) {
        return true;
    }

    let projects = getProjects();
    let projectNames = Object.keys(projects);
    return !projectNames.includes(projectName);
}


$.fn.form.settings.rules.existingPath = function (projectPath) {
    let response = checkProjectDirectory(projectPath);
    return !response.path_exists;
}


$.fn.form.settings.rules.notExistingPath = function (projectPath) {
    if (projectPath) {
        let response = checkProjectDirectory(projectPath);
        return response.path_exists;
    } else {
        return true;
    }
}


$.fn.form.settings.rules.uniquePath = function (projectPath, relocatedPath) {
    if (relocatedPath && projectPath === relocatedPath) {
        return true;
    }

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


function checkProjectDirectory(path) {
    return syncRequest('/check-existing-project', {path: path});
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
]

function assignTag(frameIdx, selectedTagIdx) {
    let tagInput = document.getElementById(frameIdx + '_tag');
    tagInput.value = selectedTagIdx;

    for (const tagIdx of [0, 1, 2]) {
        let button = document.getElementById(frameIdx + '_tag' + tagIdx);

        if (tagIdx == selectedTagIdx) {
            button.classList.add(tagColors[tagIdx]);
        } else {
            button.classList.remove(tagColors[tagIdx]);
        }
    }
}


function editClass(index, shouldEdit) {
    let classShow = $('#classShow' + index);
    let classEdit = $('#classEdit' + index);

    if (shouldEdit) {
        classShow.hide();
        classEdit.show();
    } else {
        classShow.show();
        classEdit.hide();
    }
}
