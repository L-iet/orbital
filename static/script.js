var formids = ['p2d', 'p1d', 'orb']
var choices = document.getElementById('choices');
var currChoice;

choices.addEventListener('change', function (e) {
	var relevantForm = document.getElementById(e.target.value);
	console.log(e.target.value);
	if (relevantForm.style.display === 'none'){
		for(var i=0; i<3; i++){
			if (formids[i] != e.target.value)
				document.getElementById(formids[i]).style.display = 'none';
			else
				document.getElementById(formids[i]).style.display = 'block';
		}

	}
	// else relevantForm.style.display = 'none';
});

function getCurrChoice(cc_ind) {
	currChoiceInd = cc_ind;
	document.getElementById('choices').selectedIndex = cc_ind;
}

function validateOrb() {
	let n = document.forms['orb']['n'].value;
	let l = document.forms['orb']['l'].value;
	let m = document.forms['orb']['m'].value;
	if (l >= n) {
		alert("Azimuthal quantum number must be less than Principal");
		return false;
	}
	if (Math.abs(m) > l) {
		alert("Absolute value of magnetic quantum number cannot exceed Azimuthal");
		return false;
	}

}


