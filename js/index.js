firebase.auth().onAuthStateChanged((user) => {
    if (user) {
        console.log(user);
    } else {
        console.log("not logged in!!!");
    }
});