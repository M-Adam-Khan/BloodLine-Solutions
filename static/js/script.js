function validateBasicDetails() {
    var name = document.getElementById('name').value;
    var gender = document.getElementById('gender').value;
    var age = document.getElementById('age').value;
    var weight = document.getElementById('weight').value;

    // Perform checks for age and weight
    if (age < 18 || age > 60) {
        alert("You are not eligible to donate blood due to age restrictions.");
        return;
    }
    if (weight < 45) {
        alert("You are not eligible to donate blood due to insufficient weight.");
        return;
    }

    // Hide basic details form and show health questions form
    document.getElementById('basic-details-form').style.display = "none";
    document.getElementById('health-questions-form').style.display = "block";
}

// Show further questions based on responses
document.getElementById('fever').addEventListener('input', function() {
    var fever = document.getElementById('fever').value.toLowerCase();
    if (fever === 'yes') {
        document.getElementById('fever-details').style.display = "block";
    } else {
        document.getElementById('fever-details').style.display = "none";
    }
});

document.getElementById('chronic_condition').addEventListener('input', function() {
    var chronicCondition = document.getElementById('chronic_condition').value.toLowerCase();
    if (chronicCondition === 'yes') {
        document.getElementById('chronic-condition-details').style.display = "block";
    } else {
        document.getElementById('chronic-condition-details').style.display = "none";
    }
});
