firebase.auth().onAuthStateChanged((user) => {
    if (user) {
        console.log(user);
    } else {
        console.log("not logged in!!!");
    }
});
const ctx = document.getElementById('alertsChart').getContext('2d');
const alertsChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['1', '2', '3', '4', '5', '6', '7'], // Zilele săptămânii
        datasets: [{
            label: 'Număr de alerte',
            data: [40, 67, 89, 23, 56, 78, 90], // Datele alertelor
            backgroundColor: [
                'rgba(255, 99, 132, 0.7)',  // Roșu
                'rgba(54, 162, 235, 0.7)',  // Albastru
                'rgba(255, 206, 86, 0.7)',  // Galben
                'rgba(75, 192, 192, 0.7)',  // Verde
                'rgba(153, 102, 255, 0.7)', // Mov
                'rgba(255, 159, 64, 0.7)',  // Portocaliu
                'rgba(199, 199, 199, 0.7)'  // Gri
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
            borderRadius: 5, // Colțuri rotunjite pentru bare
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
