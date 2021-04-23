function createProjectTag(projectTags) {
    let tag = document.getElementById('tag');
    let tagLabel = document.getElementById('tagLabel');
    if (tag.value == '') {
        setFormWarning(tagLabel, tag, 'Enter a proper unique tag name');
        return false;
    } else if (checkIfTagExist(projectTags, 'tag', 'tagLabel')) {
        return false;
    }
    return true;
}


function checkIfTagExist(projectTags, tagId, error) {
    let tag = document.getElementById(tagId);
    let errorLabel = document.getElementById(error);
    projectTagNames = Object.values(projectTags);

    if (projectTagNames.includes(tag.value)) {
        setFormWarning(errorLabel, tag, 'This tag name already exist');
        return true;
    }
    setFormWarning(errorLabel, tag, '');
    return false;
}


function editProjectTag(tagIdx) {
    let editTag = document.getElementById(`editTag_${tagIdx}`);
    let removeTag = document.getElementById(`removeTag_${tagIdx}`);
    let cancel = document.getElementById(`cancel_${tagIdx}`);
    let saveTag = document.getElementById(`saveTag_${tagIdx}`);
    let tag = document.getElementById(`tag_${tagIdx}`);

    tag.disabled = false;
    editTag.classList.add('uk-hidden');
    removeTag.classList.add('uk-hidden');
    cancel.classList.remove('uk-hidden');
    saveTag.classList.remove('uk-hidden');
}


async function saveProjectTag(tagIdx, url, projectTags) {
    let editTag = document.getElementById(`editTag_${tagIdx}`);
    let removeTag = document.getElementById(`removeTag_${tagIdx}`);
    let cancel = document.getElementById(`cancel_${tagIdx}`);
    let saveTag = document.getElementById(`saveTag_${tagIdx}`);
    let tag = document.getElementById(`tag_${tagIdx}`);
    let path = document.getElementById('path').value;
    let project = document.getElementById('project').value;

    data = {
        path: path,
        tagIdx: tagIdx,
        newTagName: tag.value,
    };

    if (checkIfTagExist(projectTags, `tag_${tagIdx}`, 'error')) {
        return false;
    }
    let response = await asyncRequest(url, data);

    if (response.success) {
        tag.disabled = true;
        editTag.classList.remove('uk-hidden');
        removeTag.classList.remove('uk-hidden');
        cancel.classList.add('uk-hidden');
        saveTag.classList.add('uk-hidden');
    }
}


async function removeProjectTag(tagIdx, url) {
    let project = document.getElementById('project').value;
    let path = document.getElementById('path').value;
    data = {
        path: path,
        tagIdx: tagIdx,
    };

    let response = await asyncRequest(url, data);
    if (response.success) {
        window.location.href = `/tags/${project}`;
    }
}


function cancelEditProjectTag(tagIdx) {
    let editTag = document.getElementById(`editTag_${tagIdx}`);
    let removeTag = document.getElementById(`removeTag_${tagIdx}`);
    let cancel = document.getElementById(`cancel_${tagIdx}`);
    let saveTag = document.getElementById(`saveTag_${tagIdx}`);
    let tag = document.getElementById(`tag_${tagIdx}`);
    let project = document.getElementById('project').value;
    let error = document.getElementById('error');

    setFormWarning(error, tag, '');

    tag.value = tag.attributes.value.value;
    tag.disabled = true;
    editTag.classList.remove('uk-hidden');
    removeTag.classList.remove('uk-hidden');
    cancel.classList.add('uk-hidden');
    saveTag.classList.add('uk-hidden');
}
