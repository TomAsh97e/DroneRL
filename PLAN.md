# DroneRL: plan projektu

Wersja 1.1, 2026-10-24. Etapy mają identyfikatory M0-M7, raporty etapów odwołują się do tych identyfikatorów. Szczegółowe plany etapów są w katalogu `plan/` (`plan/M0.md`, `plan/M1.md`, `plan/M2.md`) i rozwijają odpowiednie sekcje tego dokumentu.

## 1. Cel

Dron, który sam wybiera trasę lotu do zadanego celu i omija przeszkody na podstawie obrazu z kamery RGB i podstawowych czujników. Polityka sterowania jest uczona metodami RL w symulacji, a potem przenoszona na prawdziwy dron (sim2real).

Cel nadrzędny to nauka RL i ML w praktyce, dlatego:
- etapy idą od prostych zadań do trudnych i każdy wprowadza nowe zagadnienia,
- tam, gdzie ma to wartość edukacyjną, piszemy własną implementację (np. PPO), a gotowa biblioteka służy jako punkt odniesienia,
- każdy etap kończy się mierzalnym kryterium ukończenia.

Co znaczy "dobieranie trasy": lokalna, reaktywna nawigacja. Dron zna położenie celu względem siebie (np. 10 m przed startem, 2 m w lewo) i co 100 ms decyduje, jak się poruszyć, widząc przeszkody wyłącznie kamerą. Nie budujemy globalnej mapy ani klasycznego planera, choć klasyczny planer posłuży jako punkt odniesienia w symulacji.

## 2. Założenia

### 2.1 Sterowanie hierarchiczne

Najważniejsza decyzja projektu: RL nie steruje silnikami. Stabilizacją, estymacją stanu i failsafe'ami zajmuje się autopilot PX4, a polityka RL wydaje komendy prędkości, tak jak pilot drążkami w trybie Position. Powody:
- luka sim2real dla sterowania silnikami jest ogromna (aerodynamika, ESC, wibracje), dla komend prędkości jest mała,
- PX4 ma sprawdzone zabezpieczenia: utrata RC, utrata komend offboard, geofence, niska bateria,
- PX4 SITL w Gazebo uruchamia dokładnie ten sam firmware, który poleci na dronie,
- ta sama polityka zadziała na różnych platformach.

Sterowanie niskopoziomowe (obroty silników) ćwiczymy tylko w symulacji w M1, żeby zobaczyć różnicę w praktyce.

### 2.2 Percepcja oddzielona od decyzji

```
 kamera RGB                  PX4 EKF2 (IMU, flow, ToF, GPS)
     |                               |
     v                               v
 percepcja ML                stan: prędkość, wysokość,
 RGB -> głębia 48x64         wektor do celu
     |                               |
     +--------------+----------------+
                    v
            polityka RL (10 Hz)
                    |  v_fwd, v_lat, yaw_rate
                    v
           filtr bezpieczeństwa
                    |
                    v
      PX4: tryb offboard, regulatory prędkości i orientacji
                    |
                    v
                 silniki
```

Dlaczego głębia jako reprezentacja pośrednia:
- politykę trenujemy na mapach głębi, które we własnym symulatorze liczy się tanio (ray-casting na GPU, tysiące środowisk równolegle),
- głębia wygląda podobnie w symulacji i w rzeczywistości, a tekstury RGB bardzo się różnią,
- moduł RGB -> głębia to wytrenowana sieć (lidar nie jest potrzebny), którą dostrajamy i oceniamy osobno: to jest część ML projektu,
- błędy sieci głębi mierzymy na danych i odtwarzamy w symulatorze jako szum, więc polityka uczy się z nimi radzić.

Polityka end-to-end (RGB -> akcja) jest w M7 jako eksperyment porównawczy.

### 2.3 Czujniki

| Czujnik | Zastosowanie | Status |
|---|---|---|
| IMU, barometr, magnetometr | stabilizacja, EKF | w kontrolerze lotu |
| Kamera RGB z przodu, FOV 100-120° | wykrywanie przeszkód | wymagana |
| Optical flow + dalmierz ToF skierowane w dół | utrzymanie pozycji w pomieszczeniu bez GPS | wymagane do lotów w hali |
| GPS | loty na zewnątrz, geofence | w zestawie |
| Lidar 360, kamera głębi lub stereo | - | niepotrzebne |

Optical flow z ToF to tani moduł (rzędu 100-300 zł) i bez niego dron w pomieszczeniu dryfuje, bo EKF nie zna pozycji. Kamera głębi uprościłaby percepcję, ale wycięłaby z projektu najciekawszą część ML.

### 2.4 Zadanie i otoczenie

- Wersja 1: lot na stałej wysokości (ok. 1,2 m), przeszkody pionowe (słupy, pudła, ściany). Pełne 3D w M7.
- Prędkość: start od 0,5 m/s, docelowo 1-1,5 m/s.
- Przeszkody statyczne.
- Cel podany jako punkt względem startu, pozycja z EKF (flow + ToF w hali lub GPS na zewnątrz). Misje są krótkie (10-20 m), więc dryf odometrii jest akceptowalny.
- Miejsce testów: hala (najprościej pod względem prawa i pogody) albo otwarty teren w podkategorii A3.

### 2.5 Kontrakt interfejsu v1

Kontrakt jest jedynym źródłem prawdy o obserwacjach i akcjach. Żyje w kodzie (`dronerl/contract.py`) i importują go wszystkie symulatory oraz węzły ROS 2. Każda zmiana to nowa wersja kontraktu i ponowny trening. Wartości poniżej są domyślne i można je stroić w M2.

| Element | Wartość v1 |
|---|---|
| Częstotliwość polityki | 10 Hz |
| Akcja | a ∈ [-1, 1]³ skalowane do: v_fwd ∈ [-0,3; 1,0] m/s, v_lat ∈ [-0,5; 0,5] m/s, yaw_rate ∈ [-60; 60] °/s; wysokość trzyma autopilot |
| Układ odniesienia | heading frame FLU (x do przodu, y w lewo, z w górę, bez przechyleń); konwersja do NED/FRD tylko w węźle `px4_bridge` |
| Obraz | odwrotna głębia 1×48×64, FOV 90° w poziomie (wycinek z kamery szerokokątnej po rektyfikacji), zakres 0,2-8 m, wartości w [0, 1] |
| Wektor stanu | prędkość pozioma (2), yaw_rate (1), wysokość (1), kierunek do celu jako cos i sin (2), log odległości do celu (1), poprzednia akcja (3) |
| Historia | 3 ostatnie klatki lub GRU |
| Epizod | maks. 60 s; sukces: odległość od celu < 0,5 m; porażka: kolizja (odległość od przeszkody < promień drona + margines), wyjście poza obszar, limit czasu |

Nagroda startowa:

```
r_t = (d_{t-1} - d_t) + 10·[cel] - 10·[kolizja] - 0,01 - 0,05·‖a_t - a_{t-1}‖² - kara za zbliżenie do przeszkody
```

gdzie d_t to odległość do celu w metrach.

### 2.6 Bezpieczeństwo

- Pilot z aparaturą RC zawsze może przejąć sterowanie (przełącznik trybu Position / Land) i ma osobny kill switch.
- Między polityką a PX4 działa filtr bezpieczeństwa: limity prędkości, geofence, minimalna i maksymalna wysokość, watchdog (brak nowej komendy przez 300 ms oznacza zawis).
- Failsafe'y PX4: utrata offboard (`COM_OF_LOSS_T`, `COM_OBL_RC_ACT`), utrata RC, niska bateria, geofence.
- Drabina testów przed każdym nowym zachowaniem: odtworzenie nagranych danych, test bez śmigieł, zawis, lot bez przeszkód, miękkie przeszkody (karton, pianka), dopiero potem większa prędkość.
- Prawo (PL/UE): rejestracja operatora w ULC (drony.gov.pl) i szkolenie z egzaminem A1/A3 online. Samodzielnie zbudowany dron powyżej 250 g lata na zewnątrz w podkategorii A3, z dala od ludzi i zabudowy. Loty w zamkniętej hali co do zasady nie podlegają tym przepisom. Przed pierwszym lotem trzeba sprawdzić aktualne regulacje.

## 3. Technologie

### 3.1 Komputer deweloperski

Linux Mint 22.1 (baza Ubuntu 24.04), RTX 4060 8 GB, 20 wątków CPU, 31 GB RAM. Wnioski:
- GPU wystarczy do treningu małych sieci, dostrajania małego modelu głębi i własnego symulatora wektorowego na GPU,
- 8 GB VRAM to granica wymagań Isaac Sim / Isaac Lab, a przy renderowaniu kamer w wielu środowiskach naraz będzie za mało, dlatego Isaac nie jest głównym symulatorem (opcja w M7),
- baza Ubuntu 24.04 pasuje do ROS 2 Jazzy i Gazebo Harmonic, ale skrypty instalacyjne PX4 mogą nie obsługiwać Minta, więc stos PX4 + ROS 2 trzymamy w kontenerze Docker.

### 3.2 Oprogramowanie

| Obszar | Wybór | Uwagi |
|---|---|---|
| Język | Python | trening, percepcja, węzły ROS 2; C++ opcjonalnie dla węzłów ROS 2 |
| Środowisko Python | uv, Python 3.12 w venv | ta sama wersja co systemowa i wymagana przez gym-pybullet-drones; pybullet nie ma gotowych pakietów dla 3.12 i kompiluje się przy instalacji (ok. 40 s, wymaga gcc) |
| Deep learning | PyTorch z CUDA | |
| API środowisk | Gymnasium | |
| RL do nauki | własne DQN i PPO w stylu CleanRL | jeden plik na algorytm |
| RL referencyjnie | Stable-Baselines3 (M0-M1), rsl_rl lub skrl (M2+) | rsl_rl i skrl obsługują środowiska trzymające dane na GPU |
| Symulator 1 | gym-pybullet-drones | pełna dynamika kwadrokoptera (Crazyflie), M1 |
| Symulator 2 | własne środowisko wektorowe na GPU (PyTorch) | dynamika na poziomie prędkości + ray-casting głębi, M2-M3 |
| Symulator 3 | PX4 SITL + Gazebo Harmonic | ten sam firmware co dron, renderowanie RGB, walidacja sim2sim, M4 |
| Middleware | ROS 2 Jazzy, `px4_msgs`, Micro XRCE-DDS Agent | |
| Percepcja | Depth Anything V2 Small | licencja Apache-2.0 (większe warianty są niekomercyjne); alternatywy: Metric3D v2, UniDepth |
| Wdrożenie | ONNX, ONNX Runtime (PC), TensorRT FP16 (Jetson) | ONNX oddziela trening od wdrożenia, więc wersja Pythona po obu stronach nie musi się zgadzać |
| Eksperymenty | TensorBoard, opcjonalnie Weights & Biases; Optuna | |
| Konfiguracja | YAML + dataclassy | |
| Jakość kodu | git, pytest, ruff | |
| Logi lotów | rosbag2, PX4 ulog, PlotJuggler, Flight Review | |
| Stacja naziemna | QGroundControl | |

### 3.3 Sprzęt docelowy (decyzja przed M5)

**Wariant A (rekomendowany): PX4 z komputerem pokładowym**
- Holybro X500 V2 (PX4 Development Kit: rama, silniki, ESC, Pixhawk 6C, GPS). Ten sam dron jest dostępny w PX4 SITL (`gz_x500`), więc symulacja odpowiada sprzętowi.
- NVIDIA Jetson Orin Nano Super 8 GB: sieć głębi i polityka na pokładzie.
- Kamera RGB: CSI (IMX219, szeroki kąt) lub USB; kamera z global shutter (np. AR0234) jako opcja lepsza przy ruchu.
- Optical flow + ToF: Holybro H-Flow lub ARK Flow (DroneCAN, wspierane przez PX4).
- Aparatura RC (np. ELRS) z kill switchem, akumulatory 4S, ładowarka, siatka lub klatka do testów.
- Budżet orientacyjny: 6-9 tys. zł (ceny do sprawdzenia).
- JetPack 6 to Ubuntu 22.04 (natywnie ROS 2 Humble), więc na Jetsonie uruchamiamy Jazzy w kontenerze, żeby mieć tę samą dystrybucję co na PC.

**Wariant B (budżetowy, tylko w pomieszczeniu): Bitcraze Crazyflie 2.1+ / 2.1 Brushless + Flow deck + AI-deck**
- bardzo bezpieczny, ten sam model co w gym-pybullet-drones,
- kamera AI-deck ma niską rozdzielczość i jakość, obliczenia odbywają się na PC (obraz przez Wi-Fi, większe opóźnienia), udźwig minimalny,
- zamiast PX4 używamy stosu Crazyflie (cflib lub Crazyswarm2), co zmienia zakres M4 i M5,
- orientacyjnie 1,5-3 tys. zł.

Rekomendacja: wariant A. Wariant B ma sens przy ograniczonym budżecie albo gdy zależy nam na szybkim i bezpiecznym pierwszym locie. Architektura oparta na komendach prędkości działa w obu przypadkach.

## 4. Algorytmy i metody

| Metoda | Etap | Po co |
|---|---|---|
| DQN | M0 | RL oparte na wartości: replay buffer, target network |
| PPO | M0-M3, główny algorytm | stabilny on-policy, dobrze skaluje się z liczbą równoległych środowisk |
| SAC | M1 | off-policy, porównanie efektywności próbkowej z PPO |
| GAE, normalizacja obserwacji i nagród | M0+ | stabilność treningu |
| Kształtowanie nagrody | M1-M2 | wpływ nagrody gęstej i rzadkiej |
| Curriculum learning | M2 | rosnąca gęstość przeszkód |
| Nauczyciel z informacją uprzywilejowaną + uczeń (DAgger) | M2-M3 | nauczyciel widzi idealną głębię i mapę, uczeń tylko zaszumioną głębię z kamery |
| Asymetryczny actor-critic | M3, eksperyment | krytyk z informacją uprzywilejowaną jako alternatywa dla DAgger |
| Frame stacking / GRU | M3 | częściowa obserwowalność (wąskie pole widzenia) |
| Domain randomization | M3-M6 | dynamika, opóźnienia, szum czujników, szum głębi, kąt kamery |
| A* + pure pursuit | M2 | baseline i test rozwiązywalności map |
| Monokularna estymacja głębi, transfer learning | M3 | percepcja z RGB |
| Model błędu sieci głębi uczony z danych | M3 | szum w symulatorze odpowiada prawdziwym błędom percepcji |
| Destylacja wiedzy (duży model głębi -> mała CNN) | M3, opcjonalnie | szybsza inferencja na Jetsonie |
| Identyfikacja systemu (regresja na logach) | M4-M5 | dopasowanie symulatora do drona |
| TensorRT FP16 | M5 | wdrożenie na Jetsonie |

## 5. Docelowa struktura repozytorium

```
DroneRL/
  dronerl/          # pakiet Python
    contract.py     # kontrakt interfejsu
    envs/           # środowiska Gymnasium i wektorowe GPU
    agents/         # własne DQN i PPO
    utils/          # ziarna, katalogi runów, logowanie
    perception/     # model głębi, dane, model błędu
    train/          # skrypty treningowe i konfiguracje YAML
    eval/           # protokół ewaluacji i metryki
  ros2_ws/src/      # węzły ROS 2: depth, policy, safety, px4_bridge, mission
  sim/gazebo/       # światy i modele Gazebo
  deploy/           # eksport ONNX, TensorRT, skrypty Jetsona
  docker/           # obrazy PX4 + ROS 2 dla PC i Jetsona
  tests/
  runs/             # logi i checkpointy, poza gitem
```

## 6. Protokół ewaluacji

Wspólny dla wszystkich etapów od M2:
- stały zbiór testowy 200 map z ustalonymi ziarnami, nieużywany w treningu, w trzech poziomach trudności,
- metryki: success rate (SR), collision rate (CR), odsetek timeoutów, średni czas do celu, efektywność ścieżki (odległość w linii prostej / długość trasy), gładkość (średnie ‖Δa‖), opóźnienie inferencji,
- każdy wynik z co najmniej 3 ziaren treningu jako średnia ± odchylenie,
- każdy run zapisuje konfigurację, hash commita i checkpoint.

## 7. Etapy

Szacunki zakładają ok. 10-12 h pracy tygodniowo. M5 można prowadzić równolegle z M3 i M4. M5 i M6 wymagają pracy fizycznej przy dronie, a po stronie kodu obejmują skrypty, konfiguracje i analizy danych.

### M0: Fundamenty RL i warsztat (1-2 tyg.)

Cel: działające środowisko pracy i własne implementacje podstawowych algorytmów.

Szczegółowy plan: `plan/M0.md`.

Zakres:
- repozytorium git, uv + Python 3.12, PyTorch z CUDA, ruff, pytest, TensorBoard,
- własny DQN na CartPole-v1,
- własny PPO dla akcji ciągłych na Pendulum-v1 i LunarLander-v3 (`continuous=True`),
- te same zadania z SB3 jako punkt odniesienia.

Kryteria ukończenia:
- własny DQN: średni zwrot ≥ 475 na CartPole-v1 (100 epizodów),
- własny PPO: średni zwrot ≥ 200 na LunarLander-v3 continuous w ≤ 1M kroków, wynik zbliżony do SB3,
- krzywe uczenia w TensorBoard, powtarzalność przy stałym ziarnie.

Czego się uczysz: MDP, równania Bellmana, Q-learning, policy gradient, funkcja przewagi, GAE, clipping, entropia, eksploracja.

### M1: Dron w pełnej symulacji fizycznej (2-3 tyg.)

Cel: zrozumieć dynamikę drona i sprawdzić w praktyce, dlaczego wybieramy sterowanie hierarchiczne.

Szczegółowy plan: `plan/M1.md`.

Zakres:
- instalacja gym-pybullet-drones, uruchomienie przykładu z `HoverAviary`,
- własne zadanie GoalReach: lot do losowego punktu w 3D, obserwacja pełnego stanu, bez przeszkód,
- eksperyment 1: akcje niskopoziomowe (RPM) vs prędkościowe (VEL z wbudowanym PID),
- eksperyment 2: PPO vs SAC na wariancie VEL, po 3 ziarna,
- eksperyment 3: nagroda rzadka vs gęsta.

Kryteria ukończenia:
- wariant VEL: SR ≥ 95% na 100 losowych celach (promień sukcesu 0,15 m, z limitem czasu),
- wykresy i tabela z trzech eksperymentów z krótkim wnioskiem.

Czego się uczysz: dynamika kwadrokoptera, wpływ przestrzeni akcji na trudność zadania, on-policy vs off-policy, projektowanie nagrody.

### M2: Nawigacja z przeszkodami, polityka nauczyciela (3-4 tyg.)

Cel: polityka omijająca przeszkody przy założeniu idealnej percepcji.

Szczegółowy plan: `plan/M2.md`.

Zakres:
- `dronerl/contract.py` z kontraktem v1,
- własne środowisko wektorowe na GPU (PyTorch):
  - dynamika: prędkość dąży do zadanej z inercją pierwszego rzędu (stała τ), limitem przyspieszenia i opóźnieniem komend; parametry jako zakresy do randomizacji,
  - proceduralne mapy: pokój lub korytarz z walcami, prostopadłościanami i ścianami, z kontrolą rozwiązywalności (A* na siatce),
  - głębia liczona ray-castingiem (analityczne przecięcia z prymitywami), wykrywanie kolizji,
- baseline: A* + pure pursuit z pełną mapą,
- nauczyciel PPO (własny PPO z M0, sprawdzony względem rsl_rl lub skrl): mała CNN dla głębi + MLP dla stanu, obserwacja uprzywilejowana: idealna głębia + odległości do przeszkód w 360° (wirtualny lidar istnieje tylko w symulacji),
- curriculum gęstości przeszkód, strojenie hiperparametrów Optuną,
- testy: deterministyczność przy ziarnie, poprawność ray-castingu na znanych scenach, kolizje.

Kryteria ukończenia:
- przepustowość środowiska ≥ 20 tys. kroków/s na RTX 4060,
- nauczyciel: SR ≥ 90% i CR ≤ 5% na zbiorze testowym (poziom średni), porównanie z baseline A*,
- testy przechodzą.

Czego się uczysz: projektowanie środowisk RL, wektoryzacja na GPU, curriculum, strojenie hiperparametrów, informacja uprzywilejowana.

### M3: Percepcja z kamery i polityka ucznia (4-6 tyg.)

Cel: polityka działająca na głębi oszacowanej z obrazu RGB.

Zakres 3a, percepcja (ML):
- ewaluacja zero-shot Depth Anything V2 Small (wariant metryczny indoor) na zbiorach RGB-D: NYU Depth v2, opcjonalnie Hypersim lub TartanAir; metryki AbsRel i δ < 1,25 oraz opóźnienie,
- dostrajanie na wybranym zbiorze, porównanie przed i po,
- model błędu: na parach (predykcja, prawdziwa głębia) zbudować zaszumiacz głębi do symulatora (błąd skali, rozmyte krawędzie, dziury, szum zależny od odległości) i zweryfikować go statystycznie,
- eksport do ONNX,
- opcjonalnie: destylacja do małej CNN.

Zakres 3b, uczeń:
- obserwacje zgodne z kontraktem: zaszumiona głębia, zaszumiony stan, historia (stos klatek lub GRU),
- DAgger od nauczyciela z M2, następnie opcjonalnie dotrenowanie PPO,
- eksperyment: DAgger vs PPO z asymetrycznym krytykiem,
- randomizacja dynamiki, opóźnień i kąta kamery (symuluje przechylenia drona w locie).

Kryteria ukończenia:
- raport z ewaluacji modelu głębi: metryki, opóźnienie, uzasadnienie wyboru,
- uczeń osiąga SR ≥ 85% wyniku nauczyciela na zbiorze testowym z szumem,
- ablacje: z szumem i bez, z historią i bez.

Czego się uczysz: transfer learning, metryki głębi, imitation learning, DAgger, POMDP, domain randomization.

### M4: ROS 2 + PX4 SITL + Gazebo, sim2sim (3-4 tyg.)

Cel: pełny łańcuch jak na dronie, uruchomiony w innym symulatorze niż treningowy. Polityka, która nie przechodzi z jednego symulatora do drugiego, nie przejdzie też do rzeczywistości.

Zakres:
- kontener Docker: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, aktualne stabilne PX4, Micro XRCE-DDS Agent,
- światy Gazebo odwzorowujące arenę testową, model `x500` z kamerą (`gz_x500_mono_cam`),
- węzły ROS 2: `depth_node`, `policy_node` (ONNX), `safety_node`, `px4_bridge` (`OffboardControlMode` + `TrajectorySetpoint`, konwersja FLU -> NED), `mission_node` (uzbrajanie, start, misja, lądowanie),
- testy jednostkowe konwersji układów odniesienia,
- identyfikacja systemu w SITL: odpowiedź PX4 na skok zadanej prędkości, dopasowanie τ i opóźnienia, aktualizacja zakresów w symulatorze z M2,
- ewaluacja sim2sim; przy dużej luce: dane z Gazebo, aktualizacja modelu błędu, dotrenowanie ucznia DAggerem,
- nagrywanie rosbag, uruchamianie całości jednym plikiem launch.

Kryteria ukończenia:
- SR ≥ 70% w 3 światach po 20 przelotów,
- zmierzony budżet opóźnień (znacznik czasu klatki -> wysłanie komendy) ≤ 100 ms na PC,
- przetestowane zabezpieczenia: zabicie `policy_node` kończy się zawisem lub lądowaniem, przekroczenie geofence, przejęcie sterowania przez pilota.

Czego się uczysz: ROS 2, tryb offboard PX4, układy odniesienia, identyfikacja systemu, sim2sim.

### M5: Sprzęt: budowa, konfiguracja, identyfikacja (3-5 tyg., równolegle z M3-M4)

Przed startem: decyzja o wariancie sprzętu, miejscu testów i budżecie (sekcja 10).

Zakres:
- montaż, firmware PX4, kalibracja w QGroundControl, tryby lotu, kill switch, failsafe'y,
- loty ręczne w trybie Position (flow + ToF w hali, GPS na zewnątrz), autotuning regulatorów,
- Jetson: JetPack, kontener ROS 2, kamera (kalibracja w OpenCV, ekspozycja, znaczniki czasu), połączenie z Pixhawkiem (UART, uXRCE-DDS),
- lot offboard po zaprogramowanym kwadracie przez `px4_bridge` z M4, bez polityki,
- identyfikacja systemu z logów ulog: τ, opóźnienie, maksymalne przyspieszenie; porównanie z SITL i aktualizacja randomizacji,
- `depth_node` na Jetsonie z TensorRT FP16.

Kryteria ukończenia:
- zawis 60 s w trybie Position z dryfem < 0,3 m,
- kwadrat w trybie offboard przy 0,5 m/s z błędem śledzenia RMS < 0,3 m,
- model głębi na Jetsonie ≥ 10 Hz, cały łańcuch kamera -> komenda ≤ 150 ms,
- parametry z identyfikacji wprowadzone do symulatora.

Czego się uczysz: integracja sprzętu, kalibracja kamery, regresja na danych z lotu, optymalizacja inferencji.

### M6: Sim2real, loty autonomiczne (3-4 tyg.)

Cel: dron samodzielnie dolatuje do celu, omijając przeszkody.

Zakres:
- dotrenowanie ucznia z parametrami z M5 (dynamika, opóźnienia, model błędu głębi zmierzony na obrazach z prawdziwej kamery),
- drabina testów:
  1. odtworzenie nagranych danych przez cały potok i ocena sensowności komend,
  2. test bez śmigieł: przepływ komend i działanie failsafe'ów,
  3. zawis w trybie Position i przełączenie na offboard z polityką,
  4. lot do celu bez przeszkód, v_max 0,5 m/s,
  5. jedna miękka przeszkoda, potem 3-5 przeszkód,
  6. zwiększenie prędkości do 1 m/s,
- nagrywanie każdego lotu i analiza porażek.

Kryteria ukończenia:
- arena ok. 10 m z 3-5 przeszkodami: co najmniej 7 z 10 udanych przelotów bez interwencji pilota,
- porównanie metryk z symulacji i z rzeczywistości z analizą przyczyn różnic.

Czego się uczysz: sim2real w praktyce, diagnostyka, praca z danymi rzeczywistymi.

### M7: Rozszerzenia (opcjonalne, po M6)

- polityka end-to-end RGB -> akcja z mocną randomizacją wizualną i porównanie z podejściem modułowym,
- pełne 3D (v_z) i przeszkody na różnych wysokościach,
- szybszy lot (2-3 m/s),
- przelot przez bramki (drone racing),
- cel semantyczny, np. "leć do czerwonego obiektu", z detektorem obiektów (YOLO),
- model-based RL (DreamerV3) lub offline RL na zebranych logach,
- dostrajanie w rzeczywistości (residual RL) z twardymi ograniczeniami bezpieczeństwa,
- przeniesienie treningu do Isaac Lab po wymianie GPU.

## 8. Harmonogram orientacyjny

| Etap | Czas | Narastająco (tyg.) |
|---|---|---|
| M0 | 1-2 tyg. | 2 |
| M1 | 2-3 tyg. | 5 |
| M2 | 3-4 tyg. | 9 |
| M3 | 4-6 tyg. | 15 |
| M4 | 3-4 tyg. | 19 |
| M5 | 3-5 tyg., równolegle z M3-M4 | - |
| M6 | 3-4 tyg. | 23 |

Razem ok. 6 miesięcy do pierwszych autonomicznych lotów. Sprzęt warto zamówić najpóźniej po ukończeniu M2.

## 9. Ryzyka

| Ryzyko | Skutek | Środek zaradczy |
|---|---|---|
| Luka sim2real w percepcji | polityka nie widzi przeszkód | głębia jako reprezentacja pośrednia, model błędu z danych, test sim2sim w Gazebo |
| Opóźnienia (kamera, inferencja, łącze) | oscylacje, kolizje | pomiar budżetu opóźnień, randomizacja opóźnień w treningu, niskie prędkości |
| Dryf odometrii w hali | cel "przesuwa się" | krótkie misje, teksturowana podłoga pod optical flow, opcjonalnie znaczniki ArUco |
| Wolny trening | długie iteracje | własne środowisko na GPU, małe obrazy (48×64) |
| Błędy we własnym PPO | fałszywe wnioski | weryfikacja na zadaniach referencyjnych i porównanie z biblioteką |
| Rozbicie drona | koszty i przestoje | drabina testów, siatka, zapasowe śmigła, filtr bezpieczeństwa |
| Rozrost zakresu | brak postępów | kryteria ukończenia etapów, M7 dopiero po M6 |
| Niekompatybilność wersji (Mint, JetPack, ROS 2) | stracony czas | kontenery Docker, wersje zamrożone w `uv.lock`, zależności git przypięte do commitów |

## 10. Decyzje otwarte

1. Wariant sprzętu (A: PX4 X500 + Jetson, B: Crazyflie) i budżet: przed M5, najlepiej po M2.
2. Miejsce testów (hala czy otwarty teren): przed M4, bo wpływa na światy Gazebo i czujniki.
3. Tygodniowy czas pracy: wpływa na harmonogram.
4. Czy M3 obejmuje destylację modelu głębi.

## 11. Materiały

- Sutton, Barto, *Reinforcement Learning: An Introduction*, wyd. 2
- OpenAI Spinning Up, Hugging Face Deep RL Course, CleanRL
- Schulman i in. 2017, *Proximal Policy Optimization Algorithms*
- Schulman i in. 2015, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
- Haarnoja i in. 2018, *Soft Actor-Critic*
- Ross i in. 2011, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger)
- Tobin i in. 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*
- Sadeghi, Levine 2017, *CAD2RL: Real Single-Image Flight without a Single Real Image*
- Loquercio i in. 2018, *DroNet: Learning to Fly by Driving*
- Loquercio i in. 2021, *Learning High-Speed Flight in the Wild*
- Kaufmann i in. 2023, *Champion-level drone racing using deep reinforcement learning* (Nature)
- Zhang i in. 2024, *Back to Newton's Laws: Learning Vision-based Agile Flight via Differentiable Physics*
- Yang i in. 2024, *Depth Anything V2*
- Panerati i in. 2021, *Learning to Fly: a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control*
- Dokumentacja PX4: ROS 2 User Guide, Offboard Mode, Gazebo Simulation
