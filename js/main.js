const firebaseConfig = {
  apiKey: "AIzaSyBlcjBiH5utQ2fX_Qi9xmFtYTrESJRmTBg",
  authDomain: "shieldwaveai.firebaseapp.com",
  projectId: "shieldwaveai",
  storageBucket: "shieldwaveai.firebasestorage.app",
  messagingSenderId: "89383211833",
  appId: "1:89383211833:web:a5d180d3bb052735560ee8",
  measurementId: "G-L6Y6JDMMZW"
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