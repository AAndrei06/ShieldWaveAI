let email = document.getElementsByName('email')[0];
let password = document.getElementsByName('password')[0];
let submit = document.getElementsByName('submit_btn')[0];
console.log(submit);
submit.onclick = () => {
    console.log('Clikced');
    if(email.value.length < 1){
        showError('Emailul nu este introdus',true);
    }else if (password.value.length < 1){
        showError('Parola nu este introdusă',true);
    }else{
        firebase.auth().signInWithEmailAndPassword(email.value, password.value)
        .then((userCredential) => {
            let user = userCredential.user;
            let btn = document.getElementsByName('submit_btn')[0];
            btn.style.backgroundColor = '#4BB543';
            btn.innerHTML = 'Succes!!!';
            setTimeout(() => {
                window.location.href = '../index.html';
            }, 2000)
        })
    }
}
