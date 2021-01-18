
$(document).ready(function () {
    $('#pathSearch').search({
        apiSettings: {
            response: function (e) {
                var path = document.getElementById('path').value;
                var xhttp = new XMLHttpRequest();

                xhttp.open("POST", "/check-existing-project", false);
                xhttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');

                xhttp.send(JSON.stringify({path: path}));

                let response = JSON.parse(xhttp.responseText);
                pathExists = response.path_exists;
                classes = response.classes;
                subdirs = response.subdirs;

                var shouldCreateDirectory = $('#shouldCreateDirectory');
                var classList = document.getElementById('classList');
                var createProject = document.getElementById('createProject');

                results = [];
                for (const subdir of subdirs) {
                    results.push({title: subdir})
                }

                while (classList.firstChild) {
                    classList.removeChild(classList.lastChild);
                }

                for (const className of classes){
                    addClassInput(className);
                }
                addClassInput("");

                if (path && (pathExists || shouldCreateDirectory.checkbox('is checked'))) {
                    createProject.disabled = false;
                } else {
                    createProject.disabled = true;
                }

                if (pathExists) {
                    shouldCreateDirectory.checkbox('disable');
                } else {
                    shouldCreateDirectory.checkbox('enable');
                }

                return {results: results}
            }
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

    $('.haspopup').popup({
        inline: true,
        on: 'click',
        position: 'bottom right',
    });
});


function addClassInput(className) {
    var classList = document.getElementById('classList');
    var numClasses = classList.children.length;

    // Create new row
    var row = document.createElement('div');
    row.className = 'row';

    classInputGroup = createInputWithLabel('eye', 'Class', 'class' + numClasses, className, true)
    row.appendChild(classInputGroup);
    row.appendChild(document.createTextNode(' '));

    tag1InputGroup = createInputWithLabel('tag', 'Tag 1', 'class' + numClasses + '_tag1', '', false)
    row.appendChild(tag1InputGroup);
    row.appendChild(document.createTextNode(' '));

    tag2InputGroup = createInputWithLabel('tag', 'Tag 2', 'class' + numClasses + '_tag2', '', false)
    row.appendChild(tag2InputGroup);

    classList.appendChild(row);

    // Remove onclick handler on previous node
    if (numClasses > 0) {
        let previousLabeledInput = classList.children[numClasses - 1].children[0];
        let previousInput = previousLabeledInput.children[previousLabeledInput.children.length - 1];
        previousInput.removeAttribute('onfocus');
    }
}

function createInputWithLabel(icon, labelText, name, prefill, addOnFocus) {
    var inputGroup = document.createElement('div');
    inputGroup.className = 'ui labeled input';

    var label = document.createElement('div');
    label.className = 'ui label';

    var iconElement = document.createElement('i');
    iconElement.className = icon + ' icon';
    console.log(iconElement.className);

    label.appendChild(iconElement);
    label.appendChild(document.createTextNode(' ' + labelText + ' '));
    inputGroup.appendChild(label);

    var input = document.createElement('input');
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
