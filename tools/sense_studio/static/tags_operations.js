function createProjectTag(projectTags) {
    console.log('Create PROJECT TAGS');
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


function editProjectTag(tag_idx) {
    console.log('EDIT PROJECT TAGS');
    console.log(tag_idx);
}

function saveProjectTag(tag_idx) {
    console.log('Save PROJECT TAGS');
    console.log(tag_idx);
}

function removeProjectTag(tag_idx) {
    console.log('REMOVE PROJECT TAGS');
    console.log(tag_idx);
}

function closeProjectTag(tag_idx) {
    console.log('CLOSE PROJECT TAGS');
    console.log(tag_idx);
}

