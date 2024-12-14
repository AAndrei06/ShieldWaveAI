let name = document.getElementsByName('name')[0];
let email = document.getElementsByName('email')[0];
let password = document.getElementsByName('password')[0];
let confirm_password = document.getElementsByName('confirm_password')[0];
let submit = document.getElementsByName('submit_btn')[0];

submit.onclick = () => {
    
    if (name.value.length < 3){
        showError('Numele este prea scurt',false);
    }else if(!validateEmail(email.value)){
        showError('Email Invalid',false);
    }else if (password.value.length < 8){
        showError('Parola este prea scurtă',false)
    }else if (password.value != confirm_password.value){
        showError('Parolele nu se potrivesc',false);
    }else{

        firebase.auth().createUserWithEmailAndPassword(email.value, password.value)
        .then((userCredential) => {
            let user = userCredential.user;
            let date = new Date();
            console.log('Entered');
            usersDB.add({
                name: name.value,
                email: email.value,
                token: generateToken(32),
                created: date.getTime()
            }).then(() => {
                let btn = document.getElementsByName('submit_btn')[0];
                btn.style.backgroundColor = '#4BB543';
                btn.innerHTML = 'Succes!!!';
                setTimeout(() => {
                    window.location.href = '../index.html';
                }, 2000)
            });
        })
    }
}
