const firebaseConfig = {
  apiKey: "AIzaSyAPqKa2MSQEt-G0kEly-HeY2YEjFbnFpNM",
  authDomain: "megaanunt.firebaseapp.com",
  projectId: "megaanunt",
  storageBucket: "megaanunt.appspot.com",
  messagingSenderId: "1091398450368",
  appId: "1:1091398450368:web:e5cad303c33be121068125",
  measurementId: "G-8TE0GCLPKL"
};

const app = firebase.initializeApp(firebaseConfig);

const usersDB = firebase.firestore().collection("usersDB");

function validateEmail(email){

  return String(email)
      .toLowerCase()
      .match(
      /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|.(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
      );
};

function generateToken(n) {
  var chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  var token = '';
  for(var i = 0; i < n; i++) {
      token += chars[Math.floor(Math.random() * chars.length)];
  }
  return token;
}

function showError(error,connect){
  let write = document.querySelector('.title');
  let btn = document.getElementsByName('submit_btn')[0];
  write.innerHTML = error;
  write.style.color = '#cc0000';
  write.style.fontSize = '22px';
  btn.style.backgroundColor = '#cc0000';
  btn.style.fontSize = "38px";
  btn.innerHTML = '☹';
  setTimeout(() => {
    write.innerHTML = connect ? 'Conectează-te' : 'Înregistrează-te';
    write.style.color = '#454545';
    write.style.fontSize = '28px';
    btn.style.backgroundColor = '#8559ff';
    btn.style.fontSize = "20px";
    btn.innerHTML = connect ? 'Conectează-te' : 'Înregistrează-te';
  },3000);
}