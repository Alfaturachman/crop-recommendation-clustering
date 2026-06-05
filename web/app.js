// AgriAI Crop Recommendation Engine
// Loaded with scaler and K-Means model coordinates directly

const modelData = {
    features: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    scaler_mean: [
        50.551818181818184, 53.36272727272727, 48.14909090909091,
        25.616243851779544, 71.48177921778637, 6.469480065256364,
        103.46365541576817,
    ],
    scaler_scale: [
        36.90894257695227, 32.97838509495386, 50.636418345000635,
        5.062597617195944, 22.25875105745574, 0.7737617731081714,
        54.945896562329025,
    ],
    centroids: [
        [
            -0.6589754881592395, -0.2578846459622573, -0.4826492917235403,
            0.7345690338129728, 0.7289211483758873, 0.4100768027422757,
            -0.7519135132870886,
        ],
        [
            1.372246894166212, -0.7466322926920663, -0.3596046384052497,
            -0.014966038822083846, -0.5666055963756287, 0.4146343493099714,
            0.9937528165418535,
        ],
        [
            -0.2747651695760426, 0.4435252475028362, 0.6256858023096245,
            -1.3301477380039628, -2.4518002327628854, 1.2072173230593732,
            -0.42871945362115266,
        ],
        [
            -0.7998345899371099, 2.447954324844522, 2.9965690590852283,
            -0.5438501611315996, 0.8288799174610763, -0.6430476311004412,
            -0.013202525736158991,
        ],
        [
            -0.864981582417141, -1.0808763210485344, -0.3021660409778311,
            -0.5240881752126374, 0.8685760905464158, -0.19493018634679207,
            0.10666425633249353,
        ],
        [
            -0.8358095824737064, -0.27219612582665315, -0.4421069934081821,
            1.1259379624567933, -1.0165806606757843, -0.3372804614548243,
            0.26804393788200337,
        ],
        [
            1.3336077310340535, -1.0936328372909208, 0.03962934289130213,
            0.3002538546340968, 0.7777383090661192, -0.057613059756580924,
            -1.1971000262405056,
        ],
        [
            -0.786684394683564, -0.343898643421401, -0.46459516221757213,
            0.8994184353747571, -0.940629047409525, -2.0800252035256563,
            -0.31127958853125326,
        ],
        [
            -0.44772273033238513, 0.3819775963749177, -0.5650513818428067,
            0.7764592503430678, -0.3651141834539686, 0.9562187571765598,
            -0.749430466211416,
        ],
        [
            0.6827588166085499, -0.16679586618241363, -0.14731365251495104,
            -0.14778580820404966, 0.43999572965639794, 0.33170603011518984,
            1.3938485154575604,
        ],
        [
            1.364171907594464, 0.8511224911551906, 0.025323308437657295,
            0.3387750047569159, 0.39809918515261966, -0.6303449022020307,
            0.014407548884099246,
        ],
        [
            -0.8590780478620956, -1.0931717504431493, -0.5456470124202275,
            -1.4690274512407948, 0.8934377991979443, 0.8566831206241371,
            0.09851542249313532,
        ],
        [
            -0.8206320249877612, 0.4358647274919269, -0.5077965923327526,
            -1.0786782598762399, -2.017158649529694, -0.9733287168411675,
            0.24331078698326683,
        ],
        [
            -0.7735745374522682, -0.18671620971947656, -0.5584379180343009,
            0.5090053287311391, -0.7698029861220206, 2.8947825384224832,
            -0.8959787468373126,
        ],
        [
            0.7092566115469886, -0.12040721125071004, -0.14278600711667103,
            -0.3537955644935507, 0.5206477174340653, -0.12734053632487727,
            2.7110554998477365,
        ],
        [
            0.8134452238647426, -0.15268466390669788, -0.5487406092067565,
            -0.607939832191332, -0.19796395157901114, -0.30044435651504614,
            -0.3514362227143879,
        ],
        [
            -0.7169240634770331, 2.3799151277351958, 3.000643077608579,
            1.9282685757104034, 0.4768482789001038, -0.6866776311183832,
            -0.6152000315906189,
        ],
        [
            -0.7529937423500611, -1.0813387039155165, -0.36240742504699286,
            0.36766590383933806, 1.025588510862502, -0.6664590377911033,
            1.2976518862353932,
        ],
        [
            -0.0011483455389074263, 0.18009249096415983, 0.027982761677654167,
            2.5897913859865103, 0.9410056029002953, 0.37036783035924337,
            1.013539167342078,
        ],
        [
            -0.7333001111831526, 2.3910658061772287, 3.0052527091947803,
            -2.34585431244151, 0.4562596814193639, -0.5630195862319268,
            -0.6179422161700072,
        ],
        [
            -0.853624120243362, 0.36459142341315387, -0.5633117791773496,
            -0.4238274939521117, -0.4010969832908802, 0.23676685181444132,
            -0.9290462409115288,
        ],
        [
            1.8421995217520766, -0.22153629020564114, -0.559945535058479,
            -0.3140281160644573, 0.3731102569850531, 0.7392224299489575,
            -0.4281894017097098,
        ],
    ],
    cluster_crops: {
        0: ['Mungbean', 'Orange', 'Papaya'],
        1: ['Coffee'],
        2: ['Chickpea'],
        3: ['Grapes', 'Apple'],
        4: ['Pomegranate', 'Orange', 'Coconut'],
        5: ['Pigeonpeas', 'Mothbeans', 'Mango'],
        6: ['Watermelon', 'Muskmelon'],
        7: ['Pigeonpeas', 'Mothbeans', 'Mango'],
        8: ['Pigeonpeas', 'Mothbeans', 'Blackgram', 'Lentil', 'Mango'],
        9: ['Rice', 'Pigeonpeas', 'Papaya', 'Jute'],
        10: ['Banana', 'Cotton'],
        11: ['Pomegranate', 'Orange'],
        12: ['Chickpea', 'Kidneybeans', 'Pigeonpeas'],
        13: ['Mothbeans'],
        14: ['Rice', 'Papaya'],
        15: ['Maize', 'Papaya', 'Cotton'],
        16: ['Grapes'],
        17: ['Rice', 'Pigeonpeas', 'Orange', 'Coconut'],
        18: ['Papaya'],
        19: ['Grapes'],
        20: ['Pigeonpeas', 'Mothbeans', 'Blackgram', 'Lentil'],
        21: ['Maize', 'Watermelon', 'Cotton'],
    },
};

// Emojis and descriptions for crop recommendations
const cropCatalog = {
    rice: {
        icon: 'wheat',
        desc: 'Tanaman pangan utama yang membutuhkan curah hujan dan kelembaban sangat tinggi.',
    },
    maize: {
        icon: 'wheat',
        desc: 'Tanaman palawija jagung dengan kebutuhan air sedang dan tanah berdrainase baik.',
    },
    chickpea: {
        icon: 'sprout',
        desc: 'Kacang arab tangguh yang tumbuh sangat baik di iklim kering/lembab sedang.',
    },
    kidneybeans: {
        icon: 'sprout',
        desc: 'Kacang merah dengan kandungan protein tinggi yang cocok untuk tanah dingin basah.',
    },
    pigeonpeas: {
        icon: 'sprout',
        desc: 'Kacang gude tahan kekeringan yang berfungsi memperbaiki kualitas unsur Nitrogen tanah.',
    },
    mothbeans: {
        icon: 'sprout',
        desc: 'Tanaman daerah arid/kering ekstrem, sangat toleran terhadap kelangkaan air.',
    },
    mungbean: {
        icon: 'sprout',
        desc: 'Kacang hijau, tanaman berumur pendek yang membutuhkan curah hujan rendah.',
    },
    blackgram: {
        icon: 'sprout',
        desc: 'Kacang hitam tropis yang cocok ditanam pada tanah sisa persawahan padi.',
    },
    lentil: {
        icon: 'sprout',
        desc: 'Miju-miju, tanaman polong kecil nutrisi tinggi yang populer di wilayah Mediterania.',
    },
    pomegranate: {
        icon: 'apple',
        desc: 'Delima, buah eksotis yang menyukai sinar matahari penuh dan suhu hangat.',
    },
    banana: {
        icon: 'banana',
        desc: 'Pisang, tanaman buah tropis basah yang memerlukan air melimpah sepanjang tahun.',
    },
    mango: {
        icon: 'citrus',
        desc: 'Mangga, pohon buah musiman tropis yang tumbuh subur di tanah gembur.',
    },
    grapes: {
        icon: 'grape',
        desc: 'Anggur, memerlukan tanah kering berpasir dengan kandungan Kalium (K) yang sangat tinggi.',
    },
    watermelon: {
        icon: 'citrus',
        desc: 'Semangka, buah kaya air yang tumbuh optimal di tanah hangat berpasir.',
    },
    muskmelon: {
        icon: 'citrus',
        desc: 'Blewah, menyukai iklim hangat, kelembaban sedang, dan tanah kaya hara.',
    },
    apple: {
        icon: 'apple',
        desc: 'Apel, membutuhkan iklim pegunungan dingin dengan pupuk Kalium tinggi.',
    },
    orange: {
        icon: 'citrus',
        desc: 'Jeruk, pohon sitrus yang menyukai pH tanah cenderung netral dan sinar matahari.',
    },
    papaya: {
        icon: 'citrus',
        desc: 'Pepaya, tanaman buah tropis cepat tumbuh yang menyukai tanah subur gembur.',
    },
    coconut: {
        icon: 'sprout',
        desc: 'Kelapa, tanaman khas pesisir pantai dengan kebutuhan kelembaban dan air tinggi.',
    },
    cotton: {
        icon: 'cloud',
        desc: 'Kapas, tanaman serat industri yang membutuhkan iklim kering hangat dengan irigasi cukup.',
    },
    jute: {
        icon: 'leaf',
        desc: 'Yute, tanaman serat alami yang membutuhkan kondisi tropis basah atau rawa.',
    },
    coffee: {
        icon: 'coffee',
        desc: 'Kopi, tanaman perkebunan dataran tinggi yang membutuhkan suhu sejuk dan curah hujan teratur.',
    },
};

// Preset Configurations
const presets = {
    padi: {
        N: 90,
        P: 42,
        K: 43,
        temp: 21.5,
        humidity: 82.0,
        ph: 6.5,
        rainfall: 202.9,
    },
    kering: {
        N: 20,
        P: 15,
        K: 15,
        temp: 28.5,
        humidity: 40.0,
        ph: 7.2,
        rainfall: 45.0,
    },
    dingin: {
        N: 80,
        P: 120,
        K: 200,
        temp: 16.5,
        humidity: 91.0,
        ph: 5.8,
        rainfall: 110.0,
    },
    buah: {
        N: 100,
        P: 130,
        K: 200,
        temp: 22.0,
        humidity: 92.0,
        ph: 6.1,
        rainfall: 118.0,
    },
};

let myChart = null;

// Initialize App
window.addEventListener('DOMContentLoaded', () => {
    // Run initial prediction
    runInference();
});

function updateValue(param, val) {
    let suffix = '';
    if (param === 'temp') suffix = ' °C';
    else if (param === 'humidity') suffix = ' %';
    else if (param === 'rainfall') suffix = ' mm';

    document.getElementById(`val-${param}`).textContent =
        parseFloat(val).toFixed(param === 'temp' || param === 'ph' ? 1 : 0) +
        suffix;

    // Debounce/run inference
    runInference();
}

function applyPreset(key) {
    const data = presets[key];
    if (!data) return;

    // Update inputs
    document.getElementById('input-N').value = data.N;
    document.getElementById('input-P').value = data.P;
    document.getElementById('input-K').value = data.K;
    document.getElementById('input-temp').value = data.temp;
    document.getElementById('input-humidity').value = data.humidity;
    document.getElementById('input-ph').value = data.ph;
    document.getElementById('input-rainfall').value = data.rainfall;

    // Update displays
    updateValue('N', data.N);
    updateValue('P', data.P);
    updateValue('K', data.K);
    updateValue('temp', data.temp);
    updateValue('humidity', data.humidity);
    updateValue('ph', data.ph);
    updateValue('rainfall', data.rainfall);
}

function runInference() {
    // Get raw values
    const N = parseFloat(document.getElementById('input-N').value);
    const P = parseFloat(document.getElementById('input-P').value);
    const K = parseFloat(document.getElementById('input-K').value);
    const temp = parseFloat(document.getElementById('input-temp').value);
    const humidity = parseFloat(
        document.getElementById('input-humidity').value,
    );
    const ph = parseFloat(document.getElementById('input-ph').value);
    const rainfall = parseFloat(
        document.getElementById('input-rainfall').value,
    );

    const rawInputs = [N, P, K, temp, humidity, ph, rainfall];

    // 1. Scaler Transform
    const scaledInputs = rawInputs.map((val, i) => {
        return (val - modelData.scaler_mean[i]) / modelData.scaler_scale[i];
    });

    // 2. K-Means Predict (Euclidean Distance to Centroids)
    let minDistance = Infinity;
    let bestCluster = -1;

    for (let c = 0; c < modelData.centroids.length; c++) {
        const centroid = modelData.centroids[c];
        let sumSquares = 0;
        for (let f = 0; f < scaledInputs.length; f++) {
            const diff = scaledInputs[f] - centroid[f];
            sumSquares += diff * diff;
        }
        const dist = Math.sqrt(sumSquares);
        if (dist < minDistance) {
            minDistance = dist;
            bestCluster = c;
        }
    }

    // 3. Get Recommendations
    const recommendedCrops = modelData.cluster_crops[bestCluster] || [];

    // 4. Update UI
    document.getElementById('display-cluster-id').textContent = bestCluster;

    // Update badges based on cluster characteristics
    const centroidVal = modelData.centroids[bestCluster];

    // Interpret cluster centers back to original scale
    const cN =
        centroidVal[0] * modelData.scaler_scale[0] + modelData.scaler_mean[0];
    const cP =
        centroidVal[1] * modelData.scaler_scale[1] + modelData.scaler_mean[1];
    const cK =
        centroidVal[2] * modelData.scaler_scale[2] + modelData.scaler_mean[2];
    const cTemp =
        centroidVal[3] * modelData.scaler_scale[3] + modelData.scaler_mean[3];
    const cHum =
        centroidVal[4] * modelData.scaler_scale[4] + modelData.scaler_mean[4];
    const cPh =
        centroidVal[5] * modelData.scaler_scale[5] + modelData.scaler_mean[5];
    const cRain =
        centroidVal[6] * modelData.scaler_scale[6] + modelData.scaler_mean[6];

    // Determine suitablity description
    let suitText = 'Subur & Stabil';
    if (cRain > 180) suitText = 'Lahan Basah (Tinggi Air)';
    else if (cRain < 60) suitText = 'Arid / Lahan Kering';
    else if (cTemp < 20) suitText = 'Dingin Pegunungan';
    else if (cK > 150) suitText = 'Kadar Kalium Tinggi';

    document.getElementById('display-suitability').textContent = suitText;

    // Describe the cluster characteristics
    let descText = `Klaster ini didominasi oleh daerah dengan curah hujan rata-rata ${cRain.toFixed(0)} mm, temperatur ${cTemp.toFixed(1)}°C, serta kelembaban ${cHum.toFixed(0)}%. `;
    if (cN > 75)
        descText +=
            'Memiliki kandungan Nitrogen tinggi, sangat baik untuk pertumbuhan daun.';
    else if (cK > 120)
        descText +=
            'Memiliki kadar Kalium yang sangat kaya, ideal untuk jenis tanaman berumbi atau buah.';
    else descText += 'Kondisi tanah memiliki kandungan unsur hara seimbang.';

    document.getElementById('display-cluster-desc').textContent = descText;

    // Quick metrics
    document.getElementById('metric-n').textContent =
        cN > 70 ? 'Tinggi' : cN < 30 ? 'Rendah' : 'Normal';
    document.getElementById('metric-climate').textContent =
        cTemp > 28 ? 'Panas' : cTemp < 19 ? 'Sejuk' : 'Tropis Sedang';
    document.getElementById('metric-water').textContent =
        cRain > 150 ? 'Basah/Melimpah' : cRain < 70 ? 'Kering/Arid' : 'Cukup';

    // Populate crop cards
    const cropsContainer = document.getElementById('crops-container');
    cropsContainer.innerHTML = '';

    recommendedCrops.forEach((cropName) => {
        const nameLower = cropName.toLowerCase();
        const info = cropCatalog[nameLower] || {
            icon: 'sprout',
            desc: 'Tanaman pertanian bernilai ekonomi tinggi.',
        };

        // Calculate simulated fit score based on closeness to centroid
        const fitScore = Math.max(
            75,
            Math.min(99, 100 - minDistance * 8),
        ).toFixed(0);

        const cropCard = document.createElement('div');
        cropCard.className = 'crop-card';
        cropCard.innerHTML = `
      <div class="crop-card-header">
        <div class="crop-emoji"><i data-lucide="${info.icon}"></i></div>
        <h4 class="crop-title">${cropName}</h4>
      </div>
      <p>${info.desc}</p>
      <div class="suitability-score-box">
        <span>Kecocokan Lahan</span>
        <span class="score-badge">${fitScore}% Cocok</span>
      </div>
    `;
        cropsContainer.appendChild(cropCard);
    });

    if (recommendedCrops.length === 0) {
        cropsContainer.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">
        <i class="fa-solid fa-triangle-exclamation" style="font-size: 32px; color: #ffeb3b; margin-bottom: 15px;"></i>
        <p>Tidak ada tanaman yang sangat mirip di klaster ini. Coba sesuaikan parameter tanah Anda.</p>
      </div>
    `;
    }

    // Update Soil Suggestions (Actionable Insights)
    updateSoilSuggestions(rawInputs);

    // Update Chart
    updateChart(rawInputs, [cN, cP, cK, cTemp, cHum, cPh, cRain]);

    // Refresh Lucide Icons dynamically
    if (window.lucide) {
        lucide.createIcons();
    }
}

function updateChart(userInput, clusterMean) {
    // We normalize values for visual plotting comparison (0 - 100% of maximum range)
    const maxRanges = [150, 145, 205, 45, 100, 10, 300];
    const userNormalized = userInput.map(
        (val, i) => (val / maxRanges[i]) * 100,
    );
    const clusterNormalized = clusterMean.map(
        (val, i) => (val / maxRanges[i]) * 100,
    );

    const labels = [
        'Nitrogen',
        'Fosfor',
        'Kalium',
        'Suhu',
        'Kelembaban',
        'pH',
        'Hujan',
    ];

    if (myChart) {
        myChart.data.datasets[0].data = userNormalized;
        myChart.data.datasets[1].data = clusterNormalized;
        myChart.update();
    } else {
        const ctx = document.getElementById('comparisonChart').getContext('2d');
        myChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Input Tanah Anda',
                        data: userNormalized,
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
                        borderColor: '#10b981',
                        borderWidth: 2,
                        pointBackgroundColor: '#10b981',
                        pointRadius: 3,
                    },
                    {
                        label: 'Rata-rata Klaster',
                        data: clusterNormalized,
                        backgroundColor: 'rgba(148, 163, 184, 0.08)',
                        borderColor: '#64748b',
                        borderWidth: 1.5,
                        pointBackgroundColor: '#64748b',
                        pointRadius: 2,
                        borderDash: [4, 4],
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        grid: {
                            color: 'rgba(15, 23, 42, 0.06)',
                        },
                        angleLines: {
                            color: 'rgba(15, 23, 42, 0.06)',
                        },
                        ticks: {
                            display: false,
                            maxTicksLimit: 5,
                        },
                        pointLabels: {
                            color: '#334155',
                            font: {
                                size: 10,
                                family: 'Plus Jakarta Sans',
                            },
                        },
                        suggestedMin: 0,
                        suggestedMax: 100,
                    },
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#0f172a',
                            font: {
                                size: 11,
                                family: 'Plus Jakarta Sans',
                            },
                        },
                        position: 'bottom',
                    },
                },
            },
        });
    }
}

// Generate dynamic agronomic recommendations based on NPK, pH, and rainfall thresholds
function updateSoilSuggestions(inputs) {
    const [N, P, K, temp, humidity, ph, rainfall] = inputs;
    const container = document.getElementById('soil-suggestions-container');
    if (!container) return;

    container.innerHTML = '';
    const suggestions = [];

    // pH Check
    if (ph < 6.0) {
        suggestions.push({
            type: 'warning',
            icon: 'alert-triangle',
            text: `<strong>Tanah Asam (pH ${ph.toFixed(1)}):</strong> Disarankan memberikan Kapur Pertanian (Dolomit) sekitar 2-4 ton/ha untuk menetralkan pH dan mengoptimalkan penyerapan hara.`,
        });
    } else if (ph > 7.5) {
        suggestions.push({
            type: 'warning',
            icon: 'alert-triangle',
            text: `<strong>Tanah Basa (pH ${ph.toFixed(1)}):</strong> Disarankan menambahkan pupuk Amonium Sulfat (ZA) atau belerang bubuk untuk menurunkan pH tanah ke rentang optimal.`,
        });
    } else {
        suggestions.push({
            type: 'success',
            icon: 'check-circle',
            text: `<strong>Keasaman Optimal (pH ${ph.toFixed(1)}):</strong> pH tanah netral, sangat ideal bagi mikroba tanah dan penyerapan unsur hara.`,
        });
    }

    // NPK Check
    const deficiencies = [];
    if (N < 40) deficiencies.push('Nitrogen (N)');
    if (P < 30) deficiencies.push('Fosfor (P)');
    if (K < 30) deficiencies.push('Kalium (K)');

    if (deficiencies.length > 0) {
        let text = `<strong>Defisiensi Unsur ${deficiencies.join(', ')}:</strong> `;
        if (N < 40)
            text += 'Berikan pupuk Urea untuk merangsang pertumbuhan daun. ';
        if (P < 30) text += 'Gunakan SP-36 untuk memperkuat perakaran. ';
        if (K < 30)
            text +=
                'Tambahkan pupuk KCl untuk meningkatkan kualitas buah/biji. ';
        suggestions.push({
            type: 'info',
            icon: 'info',
            text: text,
        });
    } else {
        suggestions.push({
            type: 'success',
            icon: 'check-circle',
            text: '<strong>Kandungan NPK Stabil:</strong> Keseimbangan nutrisi Nitrogen, Fosfor, dan Kalium tergolong mencukupi.',
        });
    }

    // Rainfall / Irrigation Check
    if (rainfall < 80) {
        suggestions.push({
            type: 'warning',
            icon: 'droplets',
            text: `<strong>Curah Hujan Rendah (${rainfall.toFixed(0)} mm):</strong> Risiko kekeringan. Disarankan menggunakan sistem irigasi tetes atau mulsa untuk menjaga kelembaban.`,
        });
    } else if (rainfall > 200) {
        suggestions.push({
            type: 'warning',
            icon: 'droplet',
            text: `<strong>Curah Hujan Tinggi (${rainfall.toFixed(0)} mm):</strong> Risiko genangan air. Pastikan drainase lahan berfungsi optimal guna mencegah pembusukan akar.`,
        });
    }

    // Render suggestion items
    suggestions.forEach((s) => {
        const item = document.createElement('div');
        item.className = `suggestion-item ${s.type}`;
        item.innerHTML = `
            <i data-lucide="${s.icon}"></i>
            <span>${s.text}</span>
        `;
        container.appendChild(item);
    });

    if (window.lucide) {
        lucide.createIcons();
    }
}

// Compile and download land analysis report as formatted .txt file
// Compile and open a beautifully formatted print window to save as PDF
function downloadReport() {
    const btn = document.getElementById('btn-download-report');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML =
        '<i class="fa-solid fa-spinner fa-spin"></i> Menyiapkan PDF...';

    const N = parseFloat(document.getElementById('input-N').value);
    const P = parseFloat(document.getElementById('input-P').value);
    const K = parseFloat(document.getElementById('input-K').value);
    const temp = parseFloat(document.getElementById('input-temp').value);
    const humidity = parseFloat(
        document.getElementById('input-humidity').value,
    );
    const ph = parseFloat(document.getElementById('input-ph').value);
    const rainfall = parseFloat(
        document.getElementById('input-rainfall').value,
    );

    const clusterId = document.getElementById('display-cluster-id').textContent;
    const suitability = document.getElementById(
        'display-suitability',
    ).textContent;
    const description = document.getElementById(
        'display-cluster-desc',
    ).textContent;

    // Extract dynamic crop recommendations
    const cropCards = document.querySelectorAll('.crop-card');
    let cropsHTML = '';
    cropCards.forEach((card) => {
        const title = card.querySelector('.crop-title')?.textContent || '';
        const desc = card.querySelector('p')?.textContent || '';
        const score = card.querySelector('.score-badge')?.textContent || '';
        const iconHTML = card.querySelector('.crop-emoji')?.innerHTML || '';
        if (title) {
            cropsHTML += `
                <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; background: #ffffff; display: flex; gap: 12px; align-items: flex-start; box-sizing: border-box;">
                    <div style="background-color: rgba(16, 185, 129, 0.06); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.12); width: 32px; height: 32px; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; padding: 4px; box-sizing: border-box;">
                        ${iconHTML}
                    </div>
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 4px 0; font-size: 13px; font-weight: 600; color: #0f172a; font-family: 'Plus Jakarta Sans', sans-serif;">${title}</h4>
                        <p style="margin: 0 0 6px 0; font-size: 10.5px; color: #475569; line-height: 1.35; font-family: 'Plus Jakarta Sans', sans-serif;">${desc}</p>
                        <div style="font-size: 10.5px; color: #10b981; font-weight: 600; font-family: 'Plus Jakarta Sans', sans-serif;">${score}</div>
                    </div>
                </div>
            `;
        }
    });
    if (!cropsHTML) {
        cropsHTML =
            '<p style="grid-column: 1/-1; color: #64748b; font-style: italic; font-size: 12px;">Tidak ada komoditas tanaman yang direkomendasikan.</p>';
    }

    // Extract dynamic soil optimizer suggestions
    const suggestionElements = document.querySelectorAll('.suggestion-item');
    let suggestionsHTML = '';
    suggestionElements.forEach((el) => {
        const iconHTML = el.querySelector('svg')?.outerHTML || '';
        const text = el.querySelector('span')?.innerHTML || '';
        let borderCol = '#10b981';
        let bgCol = 'rgba(16, 185, 129, 0.02)';
        if (el.classList.contains('warning')) {
            borderCol = '#ef4444';
            bgCol = 'rgba(239, 68, 68, 0.02)';
        } else if (el.classList.contains('info')) {
            borderCol = '#3b82f6';
            bgCol = 'rgba(59, 130, 246, 0.02)';
        }

        if (text) {
            suggestionsHTML += `
                <div style="padding: 10px 12px; background-color: ${bgCol}; border-radius: 6px; border-left: 3px solid ${borderCol}; font-size: 11px; color: #334155; margin-bottom: 8px; line-height: 1.4; display: flex; gap: 8px; align-items: flex-start; box-sizing: border-box;">
                    <div style="color: ${borderCol}; width: 14px; height: 14px; margin-top: 2px; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
                        ${iconHTML}
                    </div>
                    <span style="flex: 1; font-family: 'Plus Jakarta Sans', sans-serif;">${text}</span>
                </div>
            `;
        }
    });
    if (!suggestionsHTML) {
        suggestionsHTML =
            '<div style="padding: 10px 12px; background-color: rgba(16, 185, 129, 0.02); border-radius: 6px; border-left: 3px solid #10b981; font-size: 11px; color: #334155; font-family: \'Plus Jakarta Sans\', sans-serif;">Kondisi lahan seimbang, tidak diperlukan tindakan khusus.</div>';
    }

    // Build the report elements in an off-screen container
    const reportContainer = document.createElement('div');
    reportContainer.style.position = 'absolute';
    reportContainer.style.left = '-9999px';
    reportContainer.style.top = '-9999px';
    reportContainer.style.width = '700px';
    reportContainer.style.padding = '35px';
    reportContainer.style.backgroundColor = '#ffffff';
    reportContainer.style.fontFamily =
        "'Plus Jakarta Sans', system-ui, -apple-system, sans-serif";
    reportContainer.style.color = '#0f172a';
    reportContainer.style.boxSizing = 'border-box';

    reportContainer.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #10b981; padding-bottom: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #0f766e, #10b981); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; font-weight: bold;">🌱</div>
                <div>
                    <h1 style="font-size: 18px; margin: 0; color: #0f172a; font-weight: 700;">AgriAI</h1>
                    <p style="font-size: 10px; color: #64748b; margin: 2px 0 0 0; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">Agricultural Decision Support System</p>
                </div>
            </div>
            <div style="text-align: right; font-size: 11px; color: #475569;">
                <p style="margin: 2px 0;"><strong>Tanggal Cetak:</strong> ${new Date().toLocaleDateString('id-ID')}</p>
                <p style="margin: 2px 0;"><strong>Model:</strong> K-Means Clustering (K=22)</p>
                <p style="margin: 2px 0;"><strong>Analisis ID:</strong> AGR-${Math.floor(100000 + Math.random() * 900000)}</p>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 20px; margin-bottom: 20px;">
            <div>
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; font-weight: 700; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 10px;">Parameter Agronomi Lahan</div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 8px; font-size: 11.5px; color: #475569;">Nitrogen (N)</td><td style="padding: 6px 8px; font-size: 11.5px; font-weight: 600; text-align: right; color: #0f172a;">${N} mg/kg</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 8px; font-size: 11.5px; color: #475569;">Fosfor (P)</td><td style="padding: 6px 8px; font-size: 11.5px; font-weight: 600; text-align: right; color: #0f172a;">${P} mg/kg</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 8px; font-size: 11.5px; color: #475569;">Kalium (K)</td><td style="padding: 6px 8px; font-size: 11.5px; font-weight: 600; text-align: right; color: #0f172a;">${K} mg/kg</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 8px; font-size: 11.5px; color: #475569;">Temperatur Rata-rata</td><td style="padding: 6px 8px; font-size: 11.5px; font-weight: 600; text-align: right; color: #0f172a;">${temp.toFixed(1)} °C</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 8px; font-size: 11.5px; color: #475569;">Kelembaban Udara</td><td style="padding: 6px 8px; font-size: 11.5px; font-weight: 600; text-align: right; color: #0f172a;">${humidity.toFixed(0)} %</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 8px; font-size: 11.5px; color: #475569;">Keasaman Tanah (pH)</td><td style="padding: 6px 8px; font-size: 11.5px; font-weight: 600; text-align: right; color: #0f172a;">${ph.toFixed(1)}</td></tr>
                    <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 8px; font-size: 11.5px; color: #475569;">Curah Hujan Tahunan</td><td style="padding: 6px 8px; font-size: 11.5px; font-weight: 600; text-align: right; color: #0f172a;">${rainfall.toFixed(0)} mm</td></tr>
                </table>
            </div>

            <div style="display: flex; flex-direction: column;">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; font-weight: 700; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 10px;">Hasil Prediksi Klaster Lahan</div>
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #10b981; border-radius: 8px; padding: 14px; flex: 1; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;">
                    <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin: 0 0 4px 0;">Klaster ${clusterId}</div>
                    <div style="background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.15); color: #10b981; padding: 3px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; display: inline-block; margin-bottom: 8px; width: fit-content;">${suitability}</div>
                    <p style="font-size: 11.5px; color: #334155; line-height: 1.45; margin: 0;">${description}</p>
                </div>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; font-weight: 700; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 10px;">Varietas Tanaman yang Direkomendasikan</div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
                ${cropsHTML}
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #475569; font-weight: 700; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 10px;">Rekomendasi Tindakan Agronomi</div>
            <div>
                ${suggestionsHTML}
            </div>
        </div>

        <div style="margin-top: 35px; display: flex; justify-content: space-between; padding: 0 40px;">
            <div style="text-align: center; font-size: 11px; color: #475569;">
                <p style="margin: 0;">Mengetahui,</p>
                <p style="font-weight: 600; margin: 2px 0 0 0;">Dosen Penguji / Asisten Praktikum</p>
                <div style="margin-top: 45px; border-top: 1px solid #cbd5e1; width: 140px;"></div>
            </div>
            <div style="text-align: center; font-size: 11px; color: #475569;">
                <p style="margin: 0;">Semarang, ${new Date().toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' })}</p>
                <p style="font-weight: 600; margin: 2px 0 0 0;">Praktikan / Mahasiswa</p>
                <div style="margin-top: 45px; border-top: 1px solid #cbd5e1; width: 140px;"></div>
            </div>
        </div>

        <div style="border-top: 1px solid #e2e8f0; padding-top: 10px; text-align: center; font-size: 9px; color: #94a3b8; margin-top: 35px;">
            <p style="margin: 2px 0;">Laporan analisis ini dihasilkan otomatis melalui sistem rekomendasi berbasis K-Means Clustering.</p>
            <p style="margin: 2px 0;">UAS Pembelajaran Mesin A11.4405 © 2026. Universitas Dian Nuswantoro.</p>
        </div>
    `;

    document.body.appendChild(reportContainer);

    // html2pdf options to create a clean A4 PDF file
    const opt = {
        margin: 10,
        filename: `Laporan_Analisis_Lahan_Klaster_${clusterId}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2.5, useCORS: true, logging: false },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
    };

    html2pdf()
        .set(opt)
        .from(reportContainer)
        .save()
        .then(() => {
            document.body.removeChild(reportContainer);
            btn.disabled = false;
            btn.innerHTML = originalText;
        })
        .catch((err) => {
            console.error(err);
            if (document.body.contains(reportContainer)) {
                document.body.removeChild(reportContainer);
            }
            btn.disabled = false;
            btn.innerHTML = originalText;
            alert('Gagal mengunduh PDF. Silakan coba lagi.');
        });
}
