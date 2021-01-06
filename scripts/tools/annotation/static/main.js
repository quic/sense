
function addClassInput() {
    var classList = document.getElementById('classList');
    var numClasses = classList.children.length;

    // Create new list item
    var node = document.createElement('LI');

    var classInput = document.createElement('input');
    classInput.type = 'text';
    classInput.name = 'class' + numClasses;
    classInput.setAttribute('onclick', 'addClassInput();');

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
    var previousNode = classList.children[numClasses - 1].children[0];
    previousNode.removeAttribute('onclick');
}
