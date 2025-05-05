<h1>Instrucțiuni de utilizare</h1>
<h3>Pasul 1</h3>
<p>Asigurațivă că aveți Python 3.10.11 instalat pe calculator, sau oricare altă versiune Python 3.10<p>
<h3>Pasul 2</h3>
<p>Descărcați repositoriul pe computerul dumnevoastră, mai bine să îl plasați pe desktop, acolo veți găsi 2 foldere, in fiecare folder creați un virtual environment cu <b>python3.10 -m venv (mlenv și webenv)</b> astfel încât să fie în același director cu requirements.txt</p>
<h3>Pasul 3</h3>
<p>Acum activați ambele medii virtuale, și rulați în același director așa: <b>python3.10 -m pip install -r requirements.txt</b>, pentru ambele medii, asta va instala tot ce este necesar</p>
<h3>Pasul 4</h3>
<p>Accesați website-ul <a href="https://shieldwave.netlify.app/">ShieldWaveAI</a> și creați un cont, după creare vedem sus un token, îl copiem și în fișierul full_program.py din folderul MachineLearning chiar sus unde este cerut la AUTH_TOKEN, acolo este unul care se incepe cu Shvoe7L4.... acolo mai este și o variabilă LIVE_KEY, mai bine o lăsați "" sau puneți acolo o cheie live de pe youtube, deoarece va porni un live pe care îl veți pune în secțiunea cu link, dar nu e obligatoriu, mai bine îl lăsați "" chiar dacă este acolo o cheie o puteți scoate și pune "".</p>
<h3>Pasul 5</h3>
<p>În fișierul startup.sh modificați calea corectă la fișiere, și îl puteți porni cu ./startup.sh sau cu start.py</p>
<h3>Pasul 6</h3>
<p>S-ar putea să apară o eroare că versiunea de numpy nu e corectă, sistemul e făcut să ruleze pe numpy==1.23.x sau 1.23.5, dacă apare eroarea, vedeți ce pachet a cauzato și instalați o versiune a pachetului care să suporte numpy 1.23.5</p>
<h2>Atenție!</h2>
<p>Asigurați-vă că aveți conectată o cameră cu microfon, sau 2 camere cu microfon dacă la variabila LIVE_KEY ați pus o cheie live, dacă este doar "" nu e nevoie de a doua cameră</p>
<h3>!Notă</h3>
<p>Proiectul este creat cu tensorflow==2.12.0, din cauza erorilor la antrenarea modelului audio create de versiunile mai noi, de asta instalez numai pachete ce suportă numpy 1.23.5, și am creat 2 medii virtuale deoarece acestă versiune tensorflow nu funcționa concomitent cu firebase_admin, ce aveau cerințe diferite ale pachetului protobuf</p>
<p>S-ar putea să funcționeze doar pe Linux deoarece Raspberry Pi folosește Linux, la fel ca calculatorul meu, nu sunt sigur dacă va merge pe Windows fișierul startup.sh, nu am încercat</p>

<p>Prezentare PPTX: <a href="https://www.canva.com/design/DAGiqCyWiAc/f9aB2ae2qspc1R2svu3PsQ/edit?utm_content=DAGiqCyWiAc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton">ShieldWavePPTX</a></p>
