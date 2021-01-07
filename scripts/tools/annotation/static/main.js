
function prefillClasses() {
    var path = document.getElementById('path').value;
    var xhttp = new XMLHttpRequest();

    let json = JSON.stringify({path: path});

    xhttp.open("POST", "/check-existing-project");
    xhttp.setRequestHeader('Content-type', 'application/json; charset=utf-8');
    xhttp.responseType = 'json';

    xhttp.send(json);

    xhttp.onload = function() {
        let response = xhttp.response;
        pathExists = response.path_exists;
        classes = response.classes;
        subdirs = response.subdirs;

        console.log(pathExists, classes, subdirs);

        var shouldCreateDirectory = document.getElementById('shouldCreateDirectory')
        var pathSuggestions = document.getElementById('pathSuggestions');
        var classList = document.getElementById('classList');
        var createProject = document.getElementById('createProject');

        while (pathSuggestions.firstChild) {
           pathSuggestions.removeChild(pathSuggestions.lastChild);
        }

        for (const subdir of subdirs) {
            var option = document.createElement('option');
            option.value = subdir;
            pathSuggestions.appendChild(option);
        }

        while (classList.firstChild) {
            classList.removeChild(classList.lastChild);
        }

        if (path && (pathExists || shouldCreateDirectory.checked)) {
            for (const className of classes){
                addClassInput(className);
            }
            addClassInput("");
            createProject.disabled = false;
        } else {
            createProject.disabled = true;
        }

        if (pathExists) {
            shouldCreateDirectory.disabled = true;
        } else {
            shouldCreateDirectory.disabled = false;
        }
    };
}


function addClassInput(className) {
    var classList = document.getElementById('classList');
    var numClasses = classList.children.length;

    // Create new list item
    var node = document.createElement('LI');

    var classInput = document.createElement('input');
    classInput.type = 'text';
    classInput.name = 'class' + numClasses;
    classInput.value = className;
    classInput.setAttribute('onfocus', 'addClassInput("");');

    var labelInput0 = document.createElement('input');
    labelInput0.type = 'text';
    labelInput0.name = 'class' + numClasses + '_label0';

    var labelInput1 = document.createElement('input');
    labelInput1.type = 'text';
    labelInput1.name = 'class' + numClasses + '_label1';

    var labelInput2 = document.createElement('input');
    labelInput2.type = 'text';
    labelInput2.name = 'class' + numClasses + '_label2';

    node.appendChild(classInput);
    node.appendChild(document.createTextNode(" "))
    node.appendChild(labelInput0);
    node.appendChild(document.createTextNode(" "))
    node.appendChild(labelInput1);
    node.appendChild(document.createTextNode(" "))
    node.appendChild(labelInput2);
    classList.appendChild(node);

    // Remove onclick handler on previous node
    if (numClasses > 0) {
        var previousNode = classList.children[numClasses - 1].children[0];
        previousNode.removeAttribute('onfocus');
    }
}
