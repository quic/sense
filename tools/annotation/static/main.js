
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

                let createProject = document.getElementById('createProject');
                if (path && !response.path_exists) {
                    createProject.disabled = false;
                    // TODO: Check if project name is provided and unique
                } else {
                    createProject.disabled = true;
                    // TODO: Show explanatory message
                }

                return {results: results}
            }
        },
        onSelect: function (selectedResult, resultList) {
            let createProject = document.getElementById('createProject');
            createProject.disabled = true;
        },
        showNoResults: false,
        cache: false
    });

    $('#pathSearchImport').search({
        apiSettings: {
            response: function (e) {
                let path = document.getElementById('path').value;
                let response = checkProjectDirectory(path);

                results = [];
                for (const subdir of response.subdirs) {
                    results.push({title: subdir})
                }

                updateClassList(response.classes);
                // TODO: Disable editing

                let importProject = document.getElementById('importProject');
                if (path && response.path_exists) {
                    importProject.disabled = false;
                } else {
                    importProject.disabled = true;
                }

                return {results: results}
            }
        },
        onSelect: function (selectedResult, resultList) {
            let importProject = document.getElementById('importProject');
            let classes = checkProjectDirectory(selectedResult.title).classes;

            importProject.disabled = false;
            updateClassList(classes);
        },
        showNoResults: false,
        cache: false
    });

    $('#shouldCreateDirectory').checkbox({
        onChecked: function () {
            var createProject = document.getElementById('createProject');
            createProject.disabled = false;
        },

        onUnchecked: function () {
            var createProject = document.getElementById('createProject');
            createProject.disabled = true;
        }
    });

    $('.hasclickpopup').popup({
        inline: true,
        on: 'click',
        position: 'bottom right',
    });

    $('.hashoverpopup').popup();
});


function checkProjectDirectory(path) {
    let xhttp = new XMLHttpRequest();

    xhttp.open("POST", "/check-existing-project", false);
    xhttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');

    xhttp.send(JSON.stringify({path: path}));

    console.log(xhttp.responseText);
    return JSON.parse(xhttp.responseText);
}


function updateClassList(classes) {
    let classList = document.getElementById('classList');

    while (classList.firstChild) {
        classList.removeChild(classList.lastChild);
    }

    for (const className of classes){
        addClassInput(className);
    }
}


function addClassInput(className) {
    let classList = document.getElementById('classList');
    let numClasses = classList.children.length;

    // Create new row
    let row = document.createElement('div');
    row.className = 'class-row';

    classInputGroup = createInputWithLabel('eye', 'Class', 'class' + numClasses, className, true)
    row.appendChild(classInputGroup);
    row.appendChild(document.createTextNode(' '));

    tag1InputGroup = createInputWithLabel('tag', 'Tag 1', 'class' + numClasses + '_tag1', '', false)
    row.appendChild(tag1InputGroup);
    row.appendChild(document.createTextNode(' '));

    tag2InputGroup = createInputWithLabel('tag', 'Tag 2', 'class' + numClasses + '_tag2', '', false)
    row.appendChild(tag2InputGroup);

    classList.appendChild(row);

    // Remove onfocus handler on previous node
    if (numClasses > 0) {
        let previousLabeledInput = classList.children[numClasses - 1].children[0];
        let previousInput = previousLabeledInput.children[previousLabeledInput.children.length - 1];
        previousInput.removeAttribute('onfocus');
    }
}

function createInputWithLabel(icon, labelText, name, prefill, addOnFocus) {
    let inputGroup = document.createElement('div');
    inputGroup.className = 'ui labeled input';

    let label = document.createElement('div');
    label.className = 'ui label';

    let iconElement = document.createElement('i');
    iconElement.className = icon + ' icon';

    label.appendChild(iconElement);
    label.appendChild(document.createTextNode(' ' + labelText + ' '));
    inputGroup.appendChild(label);

    let input = document.createElement('input');
    input.type = 'text';
    input.name = name;
    input.value = prefill;
    input.placeholder = name;

    if (addOnFocus) {
        input.setAttribute('onfocus', 'addClassInput("");');
    }

    inputGroup.appendChild(input);
    return inputGroup
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
