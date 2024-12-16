let userObject = null;
let userToken = null;

let insertAlertsDiv = document.getElementsByClassName("inserted-alerts")[0];

async function getUserData() {
    const auth = firebase.auth();
    return new Promise((resolve, reject) => {
        auth.onAuthStateChanged(async (user) => {
            if (user) {
                try {
                    const querySnapshot = await usersDB.where("ID", "==", user.uid).get();
                    if (!querySnapshot.empty) {
                        userObject = querySnapshot.docs[0].data();
                        userToken = userObject.token;
                    } else {
                        console.log("No user document found with ID:", user.uid);
                    }
                    resolve(userObject);
                } catch (error) {
                    reject(error);
                }
            } else {
                console.log("No user logged in!");
                resolve(null);
            }
        });
    });
}


getUserData()
.then((userObject) => {
    console.log("User object final:", userObject);
    document.querySelector('.plain-text-token').innerText = userToken;
    alertsDB.where("token", "==", userToken).orderBy("detection_time", "desc")
    .get()
    .then((querySnapshot) => {
        let alertsByDay = {};
        
        querySnapshot.forEach((doc) => {
            let link = doc.data().detection_type == "Video" ? "Link la video" : "Link la audio";
            let color_class = "red-class";
            let confidence = doc.data().confidence;
            
            if (confidence >= 80){
                color_class = "green-class";
            } else if (confidence < 80 && confidence >= 60){
                color_class = "yellow-class";
            } else {
                color_class = "red-class";
            }

            let seconds_time = doc.data().detection_time;
            let dateObject = new Date(seconds_time * 1000);

            let year = dateObject.getFullYear();
            let month = (dateObject.getMonth() + 1).toString().padStart(2, '0');
            let day = dateObject.getDate().toString().padStart(2, '0');
            let hours = dateObject.getHours().toString().padStart(2, '0');
            let minutes = dateObject.getMinutes().toString().padStart(2, '0');
            let seconds = dateObject.getSeconds().toString().padStart(2, '0');

            let dateKey = `${day}.${month}.${year}`;

            if (!alertsByDay[dateKey]) {
                alertsByDay[dateKey] = [];
            }

            alertsByDay[dateKey].push({
                detection_type: doc.data().detection_type,
                classification: doc.data().classification,
                confidence: doc.data().confidence,
                link: doc.data().link,
                time: `${hours}:${minutes}:${seconds}`,
                color_class: color_class
            });
        });

        let insertAlertsDiv = document.getElementsByClassName("inserted-alerts")[0];
        
        for (let dateKey in alertsByDay) {
            insertAlertsDiv.innerHTML += `
                <div class="date-formated">
                    <div class="the-line"></div>
                    <div class="actual-date">${dateKey}</div>
                </div>
            `;
            let al = 0;
            alertsByDay[dateKey].forEach(alert => {
                incrementAlertCount(alert.classification);
                al += 1;
                insertAlertsDiv.innerHTML += `
                    <div class="alert-div">
                        <div>${alert.detection_type}</div>
                        <div class="hour-of-alert">${alert.time}</div>
                        <div>${alert.classification}</div>
                        <div><div class="probability ${alert.color_class}">${alert.confidence}%</div></div>
                        <div class="link-obj"><a href="${alert.link}">${alert.detection_type == "Video" ? "Link la video" : "Link la audio"}</a></div>
                    </div>
                `;
            });
            updateChartData(String(dateKey),al);
        }
    })
    .then(() => {
        document.body.style.display = "block";
    })
    .catch((error) => {
        console.log("Error getting documents: ", error);
    });

})
.catch((error) => {
    console.error("Error:", error);
});

document.querySelector(".logout").onclick = () => {
    firebase.auth().signOut().then(() => {
        console.log("logged out");
    }).catch((error) => {
        console.log(error);
    });
}

function updateChartData(newLabels, newData) {
    alertsChart.data.labels.push(newLabels);
    alertsChart.data.datasets[0].data.push(newData)
    alertsChart.update();
}

const ctx = document.getElementById('alertsChart').getContext('2d');
const alertsChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{
            label: 'Număr de alerte',
            data: [],
            backgroundColor: [
                'rgba(255, 99, 132, 0.7)',
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 206, 86, 0.7)',
                'rgba(75, 192, 192, 0.7)',
                'rgba(153, 102, 255, 0.7)',
                'rgba(255, 159, 64, 0.7)',
                'rgba(199, 199, 199, 0.7)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)',
                'rgba(199, 199, 199, 1)'
            ],
            borderWidth: 1,
            borderRadius: 5,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {

            legend: {
                display: false,
                labels: {
                    font: {
                        size: 14,
                        family: 'Arial',
                        weight: 'bold',
                        color: '#333',
                    },
                    padding: 10
                },
                position: 'top',
            },
            title: {
                display: true,
                text: 'Numărul de alerte în ultimele 7 zile',
                font: {
                    size: 14,
                    weight: 'bold',
                },
                color: '#333',
                padding: {
                    top: 10,
                    bottom: 20
                }
            },

            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(0,0,0,0.7)',
                titleColor: '#fff',
                bodyColor: '#fff',
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Zilele săptămânii',
                    color: '#333',
                    font: {
                        size: 16,
                        weight: 'bold',
                    },
                },
                ticks: {
                    color: '#666',
                    font: {
                        size: 12,
                    },
                },
                grid: {
                    color: 'rgba(200, 200, 200, 0.3)'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Număr de alerte',
                    color: '#333',
                    font: {
                        size: 16,
                        weight: 'bold',
                    },
                },
                ticks: {
                    color: '#666',
                    font: {
                        size: 12,
                    },
                },
                grid: {
                    color: 'rgba(200, 200, 200, 0.3)'
                },
                beginAtZero: true,
            }
        }
    }
});
const ctx1 = document.getElementById('alertsChart24h').getContext('2d');
const alertsChart1 = new Chart(ctx1, {
    type: 'line',
    data: {
        labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
        datasets: [{
            data: [4, 6, 8, 5, 7, 10, 12, 15, 20, 18, 14, 10, 8, 5, 6, 9, 12, 16, 20, 18, 14, 12, 8, 5],
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 2,
            tension: 0.3,
            fill: true,
            pointBackgroundColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)',
                'rgba(199, 199, 199, 1)',
            ],
            pointRadius: 5,
            pointHoverRadius: 7,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                callbacks: {
                    label: (tooltipItem) => `Alerte: ${tooltipItem.raw}`,
                },
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                titleColor: '#fff',
                bodyColor: '#fff',
            },
            title: {
                display: true,
                text: 'Numărul de alerte pe oră (24h)',
                font: {
                    size: 14,
                    weight: 'bold',
                },
                color: '#333',
                padding: {
                    top: 10,
                    bottom: 20
                }
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Orele zilei',
                    color: '#333',
                    font: {
                        size: 16,
                        weight: 'bold',
                    },
                },
                ticks: {
                    color: '#666',
                    font: {
                        size: 12,
                    },
                },
                grid: {
                    color: 'rgba(200, 200, 200, 0.3)'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Numărul de alerte',
                    color: '#333',
                    font: {
                        size: 16,
                        weight: 'bold',
                    },
                },
                ticks: {
                    color: '#666',
                    font: {
                        size: 12,
                    },
                },
                grid: {
                    color: 'rgba(200, 200, 200, 0.3)'
                },
                beginAtZero: true,
            }
        },
        elements: {
            line: {
                backgroundColor: 'white',
            }
        }
    }
});

var ctx3 = document.getElementById('alertChartPie').getContext('2d');

var alertData = {
    labels: [],
    datasets: [{
        label: 'Număr alerte',
        data: [],
        backgroundColor: [
            'rgba(255, 87, 51, 0.7)',
            'rgba(51, 255, 87, 0.7)',
            'rgba(51, 87, 255, 0.7)',
            'rgba(255, 51, 161, 0.7)'
        ],
        borderColor: [
            'rgba(255, 87, 51, 1)',
            'rgba(51, 255, 87, 1)',
            'rgba(51, 87, 255, 1)',
            'rgba(255, 51, 161, 1)'
        ],
        borderWidth: 1
    }]
};

var alertChart = new Chart(ctx3, {
    type: 'pie',
    data: alertData,
    options: {
        responsive: true,
        plugins: {
            tooltip: {
                callbacks: {
                    label: function (tooltipItem) {
                        return tooltipItem.label + ': ' + tooltipItem.raw + ' alerte';
                    }
                }
            },
            legend: {
                display: false
            },
            datalabels: {
                anchor: 'center',
                align: 'center',
                font: {
                    weight: 'bold',
                    size: 12
                },
                color: 'white',
                formatter: (value, context) => {
                    return context.chart.data.labels[context.dataIndex] + '\n' + value;
                }
            },
            title: {
                display: true,
                text: 'Cele mai detectate alerte',
                font: {
                    size: 14,
                    weight: 'bold',
                },
                color: '#333',
                padding: {
                    top: 7,
                    bottom: 1
                }
            }
        }
    },
    plugins: [ChartDataLabels]
});

function addNewAlert(alertCategory, alertDataCount) {
    const existingIndex = alertData.labels.indexOf(alertCategory);

    if (existingIndex >= 0) {
        alertData.datasets[0].data[existingIndex] += alertDataCount;
    } else {
        alertData.labels.push(alertCategory);
        alertData.datasets[0].data.push(alertDataCount);

        let newColor = getRandomColor();
        alertData.datasets[0].backgroundColor.push(newColor);
        alertData.datasets[0].borderColor.push(newColor);
    }

    alertChart.update();
}

function incrementAlertCount(alertCategory) {
    const categoryIndex = alertData.labels.indexOf(alertCategory);
    
    if (categoryIndex !== -1) {
        alertData.datasets[0].data[categoryIndex] += 1;
    } else {
        addNewAlert(alertCategory,1);
    }

    alertChart.update();
}

function getRandomColor() {
    const r = Math.floor(Math.random() * 256);
    const g = Math.floor(Math.random() * 256);
    const b = Math.floor(Math.random() * 256);
    return `rgba(${r}, ${g}, ${b}, 0.7)`;
}

let copyBtn = document.querySelector('.pure-field');

copyBtn.onclick = function () {
    document.execCommand("copy");
}

copyBtn.addEventListener("copy", function (event) {
    event.preventDefault();
    if (event.clipboardData) {
        event.clipboardData.setData("text/plain", document.getElementsByClassName('plain-text-token')[0].innerText);
        console.log(event.clipboardData.getData("text"))
        alert('Token Copiat');
    }
});

let buttons = document.getElementsByClassName('option-btn');
let tokenField = document.querySelector('.token-field');
let pureField = document.querySelector('.pure-field');

window.addEventListener('resize', () => {
    const windowWidth = window.innerWidth;

    if (windowWidth < 380) {
        buttons[0].innerHTML = "<div><i class='fa-solid fa-trash-can'></i></div>";
        buttons[1].innerHTML = "<div><i class='fa-regular fa-circle-stop'></i></div>";
        buttons[2].innerHTML = "<div><i class='fa-solid fa-arrow-right-from-bracket'></i></div>";
    }
});

const windowWidth = window.innerWidth;

if (windowWidth < 380) {
    buttons[0].innerHTML = "<div><i class='fa-solid fa-trash-can'></i></div>";
    buttons[1].innerHTML = "<div><i class='fa-regular fa-circle-stop'></i></div>";
    buttons[2].innerHTML = "<div><i class='fa-solid fa-arrow-right-from-bracket'></i></div>";
}
