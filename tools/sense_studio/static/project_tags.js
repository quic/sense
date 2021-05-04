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
    let tagShow = document.getElementById(`tagShow${tagIdx}`);
    let tagEdit = document.getElementById(`tagEdit${tagIdx}`);

    tagEdit.classList.remove('uk-hidden');
    tagShow.classList.add('uk-hidden');
}


async function saveProjectTag(tagIdx, url, projectTags) {
    let tag = document.getElementById(`tag_${tagIdx}`);
    let tagShow = document.getElementById(`tagShow${tagIdx}`);
    let tagEdit = document.getElementById(`tagEdit${tagIdx}`);
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
        tagEdit.classList.add('uk-hidden');
        tagShow.classList.remove('uk-hidden');
        window.location.href = `/tags/${project}`;
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
    let tagShow = document.getElementById(`tagShow${tagIdx}`);
    let tagEdit = document.getElementById(`tagEdit${tagIdx}`);
    let tag = document.getElementById(`tag_${tagIdx}`);
    let project = document.getElementById('project').value;
    let error = document.getElementById('error');

    setFormWarning(error, tag, '');

    tag.value = tag.attributes.value.value;
    tagEdit.classList.add('uk-hidden');
    tagShow.classList.remove('uk-hidden');
}
