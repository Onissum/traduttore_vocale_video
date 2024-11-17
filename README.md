# Traduttore Vocale Video

Questo progetto è un'applicazione Python che utilizza strumenti open-source per tradurre sottotitoli e generare audio tradotto da video. Può trascrivere automaticamente l'audio, tradurlo in un'altra lingua e rigenerare un video con sottotitoli tradotti e un nuovo audio.

## Funzionalità
- Trascrizione automatica dell'audio in sottotitoli utilizzando **Whisper**.
- Traduzione multilingue dei sottotitoli tramite **MarianMT**.
- Correzione grammaticale delle trascrizioni con **LanguageTool**.
- Generazione di sottotitoli tradotti in un file SRT.
- Creazione di un video con sottotitoli tradotti e audio rigenerato.

## Tecnologie utilizzate
- **Python 3.10+**
- **Whisper** per la trascrizione.
- **MarianMT** per la traduzione automatica.
- **Pyttsx3** per la sintesi vocale.
- **FFmpeg** per la manipolazione audio e video.
- **MoviePy** per la gestione dei file video.

## Installazione
### Prerequisiti
- **Python 3.10+** installato.
- **FFmpeg** installato e configurato nel PATH.
- **Java** installato e configurato per l'utilizzo di LanguageTool.

### Passaggi per installare il progetto
1. Clona il repository:
   ```bash
   git clone https://github.com/Onissum/traduttore_vocale_video.git
   cd traduttore_vocale_video
   ```
2. Crea un ambiente virtuale (opzionale ma consigliato):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Per Linux/Mac
   venv\Scripts\activate    # Per Windows
   ```
3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
4. Assicurati che FFmpeg e Java siano installati e configurati correttamente:
   - [FFmpeg Download](https://ffmpeg.org/download.html)
   - [Java Download](https://www.oracle.com/java/technologies/javase-downloads.html)

## Utilizzo
1. Posiziona il video da tradurre nella directory principale del progetto o in una sottocartella.
2. Esegui il programma:
   ```bash
   python main.py
   ```
3. Segui le istruzioni sul terminale per specificare il file video di input.

### Risultato
Un file video tradotto con sottotitoli e audio tradotto sarà generato nella directory di output.

## Esempio di comando
```bash
python main.py --input input_video.mp4 --language it
```

### Parametri supportati
- `--input`: specifica il file video di input.
- `--language`: lingua target per la traduzione (es. `it` per italiano, `es` per spagnolo, ecc.).
- `--speed`: regola la velocità di sintesi vocale (opzionale).


## Traduttore Video Multiplo

Lo script `traduttore_video_multiplo.py` permette di tradurre automaticamente più video presenti in una directory. Esegue diverse operazioni in modo completamente automatizzato:
- **Estrazione dell'audio** dai video.
- **Trascrizione dei dialoghi** utilizzando Whisper.
- **Traduzione automatica** con MarianMT.
- **Correzione grammaticale** tramite LanguageTool.
- **Generazione dell'audio tradotto**.
- **Creazione di sottotitoli sincronizzati**.
- **Integrazione audio e sottotitoli nel video originale**.

### Come Usarlo
1. Posiziona i video da tradurre nella cartella `video_da_tradurre`.
2. Esegui lo script con il seguente comando:
   ```bash
   python traduttore_video_multiplo.py
   ```
3. I video tradotti, con sottotitoli e audio sincronizzati, saranno salvati nella cartella `video_tradotti`.

### Requisiti
- **Python 3.8+**
- Moduli richiesti:
  - Whisper
  - MarianMT
  - MoviePy
  - Pydub
  - Pyttsx3
  - LanguageTool-Python
- Assicurati di installare tutte le dipendenze necessarie usando il comando:
   ```bash
   pip install -r requirements.txt
   ```

### Caratteristiche Tecniche
- **Lingue Supportate:** Inglese -> Italiano (personalizzabile modificando lo script).
- **Automazione Completa:** Supporta l'elaborazione in batch di più video.
- **Facile da Integrare:** Può essere adattato per altri progetti di traduzione video.

---

### File Correlati
- `traduttore_video_multiplo.py`: Script principale per la traduzione multipla.
- `requirements.txt`: Elenco delle dipendenze necessarie.

---

### Contribuisci
Se hai suggerimenti o vuoi migliorare lo script, sentiti libero di inviare una pull request o aprire un'issue su GitHub.


## Contributi
Contributi, segnalazioni di bug e richieste di funzionalità sono benvenuti! Sentiti libero di aprire una [issue](https://github.com/Onissum/traduttore_vocale_video/issues) o un [pull request](https://github.com/Onissum/traduttore_vocale_video/pulls).

## Licenza
Questo progetto è distribuito sotto la licenza MIT. Consulta il file [LICENSE](LICENSE) per ulteriori dettagli.
