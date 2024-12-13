firebase.auth().onAuthStateChanged((user) => {
    if (user) {
        console.log(user);
    } else {
        console.log("not logged in!!!");
    }
});

document.querySelector(".logout").onclick = () => {
    firebase.auth().signOut().then(() => {
        console.log("logged out");
    }).catch((error) => {
        console.log(error);
    });
}

const ctx = document.getElementById('alertsChart').getContext('2d');
const alertsChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['1', '2', '3', '4', '5', '6', '7'],
        datasets: [{
            label: 'Număr de alerte',
            data: [40, 67, 89, 23, 56, 78, 90],
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
var alertChart = new Chart(ctx3, {
    type: 'pie',
    data: {
      labels: ['Sticlă', 'Ușă', 'Persoane', 'Pisici'],
      datasets: [{
        label: 'Număr alerte',
        data: [30, 24, 12, 17],
        backgroundColor: [
          'rgba(255, 87, 51, 0.7)',   // Geam spart
          'rgba(51, 255, 87, 0.7)',   // Uși deschise
          'rgba(51, 87, 255, 0.7)',   // Persoane detectate
          'rgba(255, 51, 161, 0.7)'   // Pisici detectate
        ],
        borderColor: [
          'rgba(255, 87, 51, 1)',   
          'rgba(51, 255, 87, 1)',   
          'rgba(51, 87, 255, 1)',   
          'rgba(255, 51, 161, 1)'   
        ],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            label: function(tooltipItem) {
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
            
        },
      }
    },
    plugins: [ChartDataLabels]
  });

let copyBtn = document.querySelector('.pure-field');

copyBtn.onclick = function() {
    document.execCommand("copy");
}

copyBtn.addEventListener("copy", function(event) {
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
    
    if (windowWidth < 1550 && windowWidth > 1250){
        buttons[0].innerHTML = "<div><i class='fa-solid fa-trash-can'></i> Șterge alertele</div>";
        buttons[1].innerHTML = "<div><i class='fa-regular fa-circle-stop'></i> Dezactivează</div>";
        buttons[2].innerHTML = "<div><i class='fa-solid fa-arrow-right-from-bracket'></i> Ieși</div>";
    }

    if (windowWidth < 380){
        buttons[0].innerHTML = "<div><i class='fa-solid fa-trash-can'></i></div>";
        buttons[1].innerHTML = "<div><i class='fa-regular fa-circle-stop'></i></div>";
        buttons[2].innerHTML = "<div><i class='fa-solid fa-arrow-right-from-bracket'></i></div>";
    }
});

const windowWidth = window.innerWidth;
    
if (windowWidth < 1550 && windowWidth > 1250){
    buttons[0].innerHTML = "<div><i class='fa-solid fa-trash-can'></i> Șterge alertele</div>";
    buttons[1].innerHTML = "<div><i class='fa-regular fa-circle-stop'></i> Dezactivează</div>";
    buttons[2].innerHTML = "<div><i class='fa-solid fa-arrow-right-from-bracket'></i> Ieși</div>";
}

if (windowWidth < 380){
    buttons[0].innerHTML = "<div><i class='fa-solid fa-trash-can'></i></div>";
    buttons[1].innerHTML = "<div><i class='fa-regular fa-circle-stop'></i></div>";
    buttons[2].innerHTML = "<div><i class='fa-solid fa-arrow-right-from-bracket'></i></div>";
}
