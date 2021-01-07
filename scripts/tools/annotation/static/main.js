
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

    var tagInput1 = document.createElement('input');
    tagInput1.type = 'text';
    tagInput1.name = 'class' + numClasses + '_tag1';

    var tagInput2 = document.createElement('input');
    tagInput2.type = 'text';
    tagInput2.name = 'class' + numClasses + '_tag2';

    node.appendChild(classInput);
    node.appendChild(document.createTextNode(" "))
    node.appendChild(tagInput1);
    node.appendChild(document.createTextNode(" "))
    node.appendChild(tagInput2);
    classList.appendChild(node);

    // Remove onclick handler on previous node
    if (numClasses > 0) {
        var previousNode = classList.children[numClasses - 1].children[0];
        previousNode.removeAttribute('onfocus');
    }
}
