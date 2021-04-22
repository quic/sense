function createProjectTag(projectTags) {
    if (tag.value == '') {
        setFormWarning(tagLabel, tag, 'Enter a proper unique tag name');
        return false;
    } else if (checkIfTagExist(projectTags)) {
        return false;
    }
    return true;
}

function checkIfTagExist(projectTags) {
    let tag = document.getElementById('tag');
    let tagLabel = document.getElementById('tagLabel');
    projectTagNames = Object.values(projectTags);

    if (projectTagNames.includes(tag.value)) {
        setFormWarning(tagLabel, tag, 'This tag name already exist');
        return true;
    }
    setFormWarning(tagLabel, tag, '');
    return false;
}


function editProjectTag(tagIdx) {
    console.log('EDIT PROJECT TAGS');
    console.log(tagIdx);
}

function saveProjectTag(tagIdx) {
    console.log('Save PROJECT TAGS');
    console.log(tagIdx);
}


async function removeProjectTag(tagIdx, url) {
    let path = document.getElementById('path').value;
    let error = document.getElementById('error');
    data = {
        path: path,
        tagIdx: tagIdx,
    };

    let response = await asyncRequest(url, data);
    if (!response.success) {
        error.innerHTML = 'Error in removing tag from project tags';
        return false;
    }
    return true;
}

function cancelEditProjectTag(tagIdx) {
    console.log('CANCEL EDIT PROJECT TAGS');
    console.log(tagIdx);
}

