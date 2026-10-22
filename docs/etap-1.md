# Etap 1. Przegląd narzędzi do rozpoznawania mowy i kryteria ich oceny

## 1. Zakres etapu

W tym etapie wybrano narzędzia ASR (Automatic Speech Recognition), które zostaną porównane na nagraniach w języku polskim, oraz określono sposób pomiaru jakości transkrypcji. Porównujemy cztery narzędzia w dwóch grupach:

- modele uruchamiane lokalnie, z otwartymi wagami, które można postawić na własnym sprzęcie (GPU) bez wysyłania nagrań na zewnątrz: Whisper large-v3 i Parakeet-TDT-0.6b-v3,
- usługi chmurowe dostępne przez API, rozliczane za czas nagrania, w których model jest czarną skrzynką: Google Cloud Speech-to-Text (Chirp) i Amazon Transcribe.

Podział ma znaczenie praktyczne. W modelach lokalnych kontrolujemy wersję modelu, parametry dekodowania i przetwarzanie wstępne, ale odpowiadamy za infrastrukturę i wydajność. W usługach API jakość zależy od dostawcy, model może się zmienić bez naszej wiedzy, a nagrania opuszczają naszą infrastrukturę, co dla części zastosowań (np. dane medyczne, rozmowy z klientami) jest istotnym ograniczeniem.

## 2. Narzędzia

### 2.1. Modele lokalne

#### Whisper large-v3 (OpenAI)

Whisper to model typu encoder-decoder oparty na architekturze Transformer, opublikowany przez OpenAI w 2022 roku (Radford i in., *Robust Speech Recognition via Large-Scale Weak Supervision*). Na wejściu przyjmuje spektrogram melowy z 30-sekundowego okna audio, a dekoder generuje tekst token po tokenie, tak jak model językowy. Jeden model obsługuje rozpoznawanie mowy w 99 językach, identyfikację języka, tłumaczenie na angielski i generowanie znaczników czasu. Zadanie wybiera się specjalnymi tokenami na początku sekwencji dekodera.

Wersja large-v3 (listopad 2023) ma ok. 1,55 mld parametrów. W porównaniu z large-v2 zwiększono liczbę pasm melowych z 80 do 128, a zbiór treningowy rozszerzono do ok. 1 mln godzin nagrań ze słabymi etykietami i ok. 4 mln godzin nagrań opisanych automatycznie przez large-v2 (pseudo-labeling).

Znane wady Whispera, istotne przy ocenie:

- halucynacje na fragmentach ciszy lub szumu (model potrafi „dopisać” zdanie, którego nikt nie wypowiedział, albo zapętlić powtórzenie),
- przetwarzanie dłuższych nagrań w kolejnych 30-sekundowych oknach, co przy złym podziale może gubić słowa na granicach okien,
- brak natywnego trybu strumieniowego, więc w interfejsie konwersacyjnym trzeba go emulować krótkimi fragmentami,
- autoregresyjny dekoder z 32 warstwami, który jest wąskim gardłem przy inferencji i sprawia, że model jest wolniejszy od pozostałych lokalnych rozwiązań.

Model uruchamia się przez bibliotekę `openai-whisper`, `transformers` (Hugging Face) albo przez szybsze implementacje: `faster-whisper` (CTranslate2) i `whisper.cpp`. `faster-whisper` ma wbudowany filtr VAD (Silero), który ogranicza halucynacje na ciszy. Parametr `initial_prompt` pozwala podać fragment tekstu, który ukierunkowuje dekoder, np. na słownictwo dziedzinowe. Licencja: MIT.

#### Parakeet-TDT-0.6b-v3 (NVIDIA NeMo)

Parakeet-TDT-0.6b-v3 (sierpień 2025, ok. 600 mln parametrów) pochodzi z NeMo, frameworka NVIDIA do trenowania i uruchamiania modeli mowy. Model obsługuje 25 języków europejskich, w tym polski. Został wytrenowany głównie na zbiorze Granary, zawierającym ok. miliona godzin pseudo-etykietowanych nagrań w językach europejskich, uzupełnionym danymi własnymi NVIDIA.

Enkoder to FastConformer, czyli Conformer z agresywniejszym podpróbkowaniem sygnału na wejściu (8-krotnym zamiast 4-krotnego), co zmniejsza długość sekwencji i koszt obliczeń. Dekoder to TDT (Token-and-Duration Transducer), odmiana modelu RNN-T, w której sieć oprócz tokenu przewiduje też, o ile ramek akustycznych można przeskoczyć. Dzięki temu dekodowanie nie musi odwiedzać każdej ramki i jest wielokrotnie szybsze niż w Whisperze. Architektura transducera dobrze nadaje się też do pracy strumieniowej, co jest istotne dla interfejsu konwersacyjnego.

Model sam wykrywa język, zwraca tekst z interpunkcją i wielkimi literami oraz znaczniki czasu na poziomie słów. Według dokumentacji przetwarza w jednym przebiegu nagrania do ok. 24 minut (przy pełnej uwadze), a w trybie uwagi lokalnej znacznie dłuższe. W odróżnieniu od Whispera nie ma mechanizmu podpowiadania słownictwa przez prompt.

Model uruchamia się przez pakiet `nemo_toolkit[asr]`. W praktyce wymaga GPU NVIDIA, a na CPU działa wolno. Licencja: CC-BY-4.0.

### 2.2. Usługi API

#### Google Cloud Speech-to-Text (modele Chirp)

Speech-to-Text to usługa Google Cloud. Modele Chirp są dostępne w wersji V2 API. Wywodzą się z Universal Speech Model (USM, Zhang i in., 2023), modelu o ok. 2 mld parametrów, wstępnie trenowanego na ok. 12 mln godzin nieetykietowanego audio w ponad 300 językach i dostrajanego na mniejszych zbiorach etykietowanych. Kolejne generacje (Chirp, Chirp 2, Chirp 3) dodawały m.in. znaczniki czasu słów, tryb strumieniowy, automatyczne wykrywanie języka i diaryzację mówców. Zakres funkcji różni się między generacjami i regionami, więc przed testami trzeba go sprawdzić w dokumentacji.

API udostępnia trzy tryby: rozpoznawanie synchroniczne dla krótkich nagrań (do ok. minuty), wsadowe dla długich plików z Cloud Storage oraz strumieniowe przez gRPC. Konfigurację (model, język, opcje) zapisuje się w obiekcie `Recognizer`. Mechanizm adaptacji mowy (phrase sets, custom classes) pozwala podbić prawdopodobieństwo wskazanych fraz, co ma bezpośredni wpływ na rozpoznawanie nazw własnych (p. 3.3). Dostępność adaptacji dla poszczególnych modeli Chirp również należy zweryfikować.

#### Amazon Transcribe

Amazon Transcribe to usługa ASR w AWS, obsługująca język polski (`pl-PL`) w trybie wsadowym i strumieniowym. W trybie wsadowym nagranie wskazuje się w S3, a wynik w JSON zawiera transkrypcję, znaczniki czasu i poziom pewności każdego słowa. Tryb strumieniowy działa przez HTTP/2 lub WebSocket i zwraca wyniki częściowe, które są stopniowo stabilizowane. Ma to znaczenie dla opóźnień w dialogu.

Z punktu widzenia projektu najważniejsze funkcje to:

- custom vocabulary, czyli lista słów i fraz (np. nazwisk, nazw produktów) z opcjonalną wymową, która poprawia ich rozpoznawanie,
- vocabulary filtering, czyli maskowanie lub usuwanie wybranych słów,
- diaryzacja mówców i identyfikacja kanałów (np. osobno klient i konsultant w nagraniu stereo),
- automatyczna identyfikacja języka.

Custom language models (dostrajanie modelu językowego na własnych tekstach) i redakcja danych osobowych (PII) są dostępne tylko dla części języków, a polski nie był dotąd na tej liście. Amazon nie publikuje szczegółów architektury. Wiadomo jedynie, że od 2023 roku usługa korzysta z dużego wielojęzycznego modelu bazowego. Rozliczenie jest sekundowe, z minimalną opłatą za każde zlecenie.

### 2.3. Zestawienie

| Narzędzie | Dostawca | Architektura | Rozmiar | Strumieniowanie | Podpowiadanie słownictwa | Dostęp / licencja |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper large-v3 | OpenAI | encoder-decoder Transformer | 1,55 mld | nie (emulacja) | `initial_prompt` | lokalnie, MIT |
| Parakeet-TDT-0.6b-v3 | NVIDIA | FastConformer + TDT | 0,6 mld | tak | brak | lokalnie, CC-BY-4.0 |
| Speech-to-Text (Chirp) | Google | rodzina USM | nie podano | tak | adaptacja mowy | API, płatne |
| Amazon Transcribe | AWS | nie podano | nie podano | tak | custom vocabulary | API, płatne |

## 3. Kryteria oceny

Do porównania wybrano trzy miary: WER, CER i dokładność rozpoznawania encji. Wszystkie trzy da się policzyć automatycznie i powtarzalnie: porównują hipotezę (wynik systemu) z ręcznie przygotowaną transkrypcją wzorcową (referencją) według stałych reguł, bez oceny przez człowieka na etapie pomiaru. WER i CER opisują ogólną jakość transkrypcji, a dokładność encji mierzy to, co w interfejsie konwersacyjnym decyduje o poprawnym działaniu systemu.

WER i CER opierają się na odległości edycyjnej Levenshteina, czyli minimalnej liczbie operacji podstawienia, usunięcia i wstawienia, która przekształca jedną sekwencję w drugą. Różnią się jednostką porównania: słowo albo znak.

### 3.1. WER (Word Error Rate)

```text
WER = (S + D + I) / N
```

gdzie:

- S – liczba podstawień (słowo z referencji zastąpione innym),
- D – liczba usunięć (słowo z referencji pominięte w hipotezie),
- I – liczba wstawień (słowo w hipotezie, którego nie ma w referencji),
- N – liczba słów w referencji.

Wartości S, D i I wyznacza się z optymalnego dopasowania obu sekwencji słów. Przykład: referencja „umów wizytę na jutro rano” (N = 5), hipoteza „umów wizyta na jutro”. Dopasowanie daje jedno podstawienie (wizytę → wizyta) i jedno usunięcie (rano), czyli WER = 2/5 = 40%.

WER jest metryką podstawową, bo pozwala porównać wyniki z literaturą i benchmarkami (FLEURS, Common Voice), gdzie raportuje się właśnie WER. Trzeba pamiętać o kilku jego właściwościach:

- WER może przekroczyć 100%, bo liczba wstawień nie jest ograniczona. Dzieje się tak np. przy halucynacjach Whispera na ciszy.
- Każdy błąd waży tyle samo: pomyłka w spójniku „i” kosztuje tyle co pomyłka w nazwisku, które niesie całą informację. Tę lukę wypełnia miara z p. 3.3.
- Wynik silnie zależy od normalizacji tekstu (p. 3.4). Bez niej „Warszawa.” i „warszawa” to dwa różne słowa.
- Wynik dla całego zbioru liczy się jako sumę błędów podzieloną przez sumę słów we wszystkich nagraniach, a nie jako średnią WER z poszczególnych nagrań. Inaczej krótkie wypowiedzi mają nieproporcjonalnie duży wpływ na wynik.

Oprócz samej wartości WER raportujemy rozbicie na S, D i I, bo pokazuje ono charakter błędów danego systemu: dużo usunięć przy cichych lub zaszumionych nagraniach, dużo wstawień przy halucynacjach.

### 3.2. CER (Character Error Rate)

CER liczy się tym samym wzorem co WER, ale na poziomie znaków: S, D i I to operacje na znakach, a N to liczba znaków referencji. Spacje traktuje się jako znaki, dzięki czemu błędy sklejenia lub rozdzielenia wyrazów też są widoczne.

Dla polskiego CER jest szczególnie ważny ze względu na fleksję. Rzeczownik ma do 7 form przypadków w dwóch liczbach, czasownik odmienia się przez osoby, rodzaje i aspekty, a różnice między formami to często jedna lub dwie litery końcówki. Dla WER błędna końcówka oznacza cały błędny wyraz, choć system rozpoznał poprawnie rdzeń, a więc to, co zostało powiedziane.

Przykład: referencja „spotkałem się z nim przed domem”, hipoteza „spotkałem się z nim przed domu”.

- WER: jedno podstawienie na 6 słów, czyli ok. 16,7%.
- CER: „domem” → „domu” to jedno podstawienie (e → u) i jedno usunięcie (m), czyli 2 operacje na 31 znaków (ze spacjami), ok. 6,5%.

Oba wyniki opisują ten sam błąd, ale WER sugeruje, że system pomylił się w co szóstym słowie, a CER lepiej oddaje, jak blisko poprawnej transkrypcji był wynik. Z tego powodu raportujemy obie metryki. Duża różnica między WER a CER dla danego systemu wskazuje, że jego błędy to głównie pomyłki w końcówkach, a nie w całych wyrazach.

### 3.3. Dokładność rozpoznawania encji

W interfejsie konwersacyjnym nie wszystkie słowa są równie ważne. W wypowiedzi „przełóż spotkanie z panem Wiśniewskim na czwartek na piętnastą” system dialogowy potrzebuje trzech informacji: kto (Wiśniewski), kiedy (czwartek), o której (15:00). Błąd w słowie „pan” nie zmienia działania systemu, a błąd w nazwisku oznacza, że akcja zostanie wykonana źle lub wcale. WER i CER traktują oba błędy jednakowo, dlatego potrzebna jest osobna miara.

W referencjach oznaczamy encje istotne dla zadania:

- nazwy osób, firm i miejsc,
- daty i godziny,
- liczby, kwoty, numery telefonów, kody i numery dokumentów,
- słowa kluczowe dziedziny (np. nazwy usług lub produktów, polecenia sterujące).

```text
dokładność encji = (liczba poprawnie rozpoznanych encji) / (liczba encji w referencjach)
```

Żeby miara była obiektywna, o poprawności encji decyduje dopasowanie wyznaczone przy liczeniu WER, a nie ocena człowieka. Encja jest uznana za rozpoznaną poprawnie, jeśli wszystkie jej słowa zostały w dopasowaniu oznaczone jako zgodne, czyli nie wystąpiło w jej obrębie żadne podstawienie ani usunięcie. Porównanie odbywa się po tej samej normalizacji co dla WER i CER (p. 3.4), więc „15:00” i „piętnastą” są sprowadzane do wspólnej postaci, zanim zostaną porównane. Błędna forma fleksyjna („Wiśniewskiego” zamiast „Wiśniewskim”) liczy się jako błąd. Rozluźnienie tej reguły wymagałoby lematyzacji, która sama wprowadza błędy i uzależnia wynik od użytego narzędzia.

Oznaczenia encji w referencjach przygotowuje się raz, przed uruchomieniem testów, i są one wspólne dla wszystkich systemów.

Ta miara jest najbardziej czuła na mechanizmy podpowiadania słownictwa: adaptację mowy w Google, custom vocabulary w Amazon Transcribe i `initial_prompt` w Whisperze. Testy z podpowiedziami i bez nich pozwolą ocenić, ile każde z tych narzędzi zyskuje na dostosowaniu do dziedziny. Parakeet takiego mechanizmu nie ma, więc w tym wariancie służy jako punkt odniesienia.

### 3.4. Normalizacja tekstu przed pomiarem

Narzędzia różnią się formatem wyjścia: część zwraca tekst z interpunkcją i wielkimi literami, a liczby są zapisywane raz cyframi, raz słownie. Bez ujednolicenia porównanie mierzyłoby różnice w formatowaniu, a nie w rozpoznawaniu mowy. Przed obliczeniem wszystkich trzech miar hipotezy i referencje przechodzą tę samą normalizację:

1. zamiana na małe litery,
2. usunięcie znaków interpunkcyjnych (z wyjątkiem tych, które są częścią zapisu, np. dwukropek w godzinie przed jej konwersją),
3. ujednolicenie białych znaków,
4. zamiana liczebników zapisanych słownie na cyfry (lub odwrotnie, ale jednakowo dla wszystkich systemów),
5. zachowanie polskich znaków diakrytycznych: „zle” zamiast „źle” liczy się jako błąd, bo w języku polskim zmienia słowo lub czyni je niepoprawnym.

Normalizator dołączony do Whispera (`EnglishTextNormalizer`) jest przygotowany pod angielski i nie nadaje się do polskiego, dlatego reguły implementujemy sami. Do obliczania WER i CER oraz wyznaczenia dopasowania słów używamy biblioteki `jiwer`.

## Źródła

- A. Radford i in., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022, arXiv:2212.04356
- Karty modeli Hugging Face: `openai/whisper-large-v3`, `nvidia/parakeet-tdt-0.6b-v3`
- Y. Zhang i in., *Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages*, 2023, arXiv:2303.01037
- Dokumentacja Google Cloud Speech-to-Text V2: <https://cloud.google.com/speech-to-text/v2/docs>
- Dokumentacja Amazon Transcribe: <https://docs.aws.amazon.com/transcribe/>
