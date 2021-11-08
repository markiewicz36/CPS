import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import PySimpleGUI as sg

class Sygnal:
    def __init__(self, dane, probki, parametry, typ_sygnalu, rodzaj):
        self.dane = dane
        self.probki_czasowe = probki
        self.parametry = parametry
        self.typ = typ_sygnalu
        self.rodzaj = rodzaj

    def wykresy(self):
        plt.figure(1, figsize=(10, 5))
        plt.subplot(211)
        plt.plot(self.probki_czasowe, self.dane,
                  color='indigo') if self.rodzaj == 'ciagly' else plt.scatter(self.probki_czasowe, self.dane, s=1)
        plt.subplot(212)
        plt.hist(self.dane, 20, edgecolor='black')
        plt.show()
        plt.close()

    def wynik_okno(self):
        sg.theme('DarkBrown')  # Wygląd GUI
        # Layout
        layout = [
            [sg.Text('wartość średnia sygnału '), sg.Text(Sygnal.get_srednia(self))],
            [sg.Text('wartość bezwzględna sygnału'), sg.Text(Sygnal.get_srednia_bezwzgledna(self))],
            [sg.Text('moc średnia sygnału'), sg.Text(Sygnal.get_srednia_moc_sygnalu(self))],
            [sg.Text('wariancja wygnału'), sg.Text(Sygnal.get_wariancja(self))],
            [sg.Text('wartość skuteczna sygnału'), sg.Text(Sygnal.get_wartosc_skuteczna_sygnalu(self))],
            [sg.Button('Ok')]]
        # Create the Window
        window = sg.Window('Generator sygnału', layout)
        while True:
            event, values = window.read()
            Sygnal.wykresy(self)
            if event == sg.WIN_CLOSED or event == 'Ok':
                break
        window.close()

    def zapisz_do_pliku(self, path):
        with open(path, 'bw+') as file:
            pickle.dump(self, file)
        #Moduł pickle implementuje binarne protokoły do serializacji i de-serializacji struktury obiektowej Pythona.

    @staticmethod
    def wczytaj_z_pliku(path):
        try:
            file = open(path, 'rb')
            return pickle.load(file)
        except:
            return None

    def get_srednia(self):
        suma = 0
        for index in range(len(self.dane)):
            suma += self.dane[index]

        return suma / len(self.dane)

    def get_srednia_bezwzgledna(self):
        suma = 0
        for index in range(len(self.dane)):
            suma += np.abs(self.dane[index])
        return suma / len(self.dane)

    def get_srednia_moc_sygnalu(self):
        suma = 0
        for index in range(len(self.dane)):
            suma += np.power(self.dane[index], 2)
        return suma / len(self.dane)

    def get_wariancja(self):
        suma = 0
        mean = self.get_srednia()
        for index in range(len(self.dane)):
            suma += np.power(self.dane[index] - mean, 2)
        return suma / len(self.dane)

    def get_wartosc_skuteczna_sygnalu(self):
        return math.sqrt(np.abs(self.get_srednia_moc_sygnalu()))

    @staticmethod
    def dodawanie(pierwszy_sygnal, drugi_sygnal):
        czas_poczatkowy = min(pierwszy_sygnal.parametry['czas_poczatkowy'], drugi_sygnal.parametry['czas_poczatkowy'])

        czas_koncowy = max(pierwszy_sygnal.parametry['czas_poczatkowy'] + pierwszy_sygnal.parametry['czas_trwania'],
                       drugi_sygnal.parametry['czas_poczatkowy'], drugi_sygnal.parametry['czas_trwania'])
        all_probki = np.arange(czas_poczatkowy, czas_koncowy, 1 / pierwszy_sygnal.parametry['czestotliwosc_probkowania'])

        dane_nowy_sygnal = []
        tmp_a = 0
        tmp_b = 0
        for probka in all_probki:
            tmp_a_index = np.where(pierwszy_sygnal.probki_czasowe == probka)[0][
                0] if probka in pierwszy_sygnal.probki_czasowe else None
            tmp_b_index = np.where(drugi_sygnal.probki_czasowe == probka)[0][
                0] if probka in drugi_sygnal.probki_czasowe else None
            tmp_a = pierwszy_sygnal.dane[tmp_a_index] if tmp_a_index is not None else 0
            tmp_b = drugi_sygnal.dane[tmp_b_index] if tmp_b_index is not None else 0
            dane_nowy_sygnal.append(tmp_a + tmp_b)
        parametry = {
            'czestotliwosc_probkowania': pierwszy_sygnal.parametry['czestotliwosc_probkowania'],
            'czas_trwania': czas_koncowy - czas_poczatkowy,
            'czas_poczatkowy': czas_poczatkowy
        }
        return Sygnal(dane_nowy_sygnal, all_probki, parametry, 'ciagly', 'ciagly')

    @staticmethod
    def mnozenie(pierwszy_sygnal, drugi_sygnal):
        czas_poczatkowy = min(pierwszy_sygnal.parametry['czas_poczatkowy'], drugi_sygnal.parametry['czas_poczatkowy'])
        czas_koncowy = max(pierwszy_sygnal.parametry['czas_poczatkowy'] + pierwszy_sygnal.parametry['czas_trwania'],
                       drugi_sygnal.parametry['czas_poczatkowy'], drugi_sygnal.parametry['czas_trwania'])
        all_probki = np.arange(czas_poczatkowy, czas_koncowy, 1 / pierwszy_sygnal.parametry['czestotliwosc_probkowania'])
        dane_nowy_sygnal = []
        tmp_a = 0
        tmp_b = 0
        for probka in all_probki:
            tmp_a_index = np.where(pierwszy_sygnal.probki_czasowe == probka)[0][
                0] if probka in pierwszy_sygnal.probki_czasowe else None
            tmp_b_index = np.where(drugi_sygnal.probki_czasowe == probka)[0][
                0] if probka in drugi_sygnal.probki_czasowe else None
            tmp_a = pierwszy_sygnal.dane[tmp_a_index] if tmp_a_index is not None else 1
            tmp_b = drugi_sygnal.dane[tmp_b_index] if tmp_b_index is not None else 1
            dane_nowy_sygnal.append(tmp_a * tmp_b)
        parametry = {
            'czestotliwosc_probkowania': pierwszy_sygnal.parametry['czestotliwosc_probkowania'],
            'czas_trwania': czas_koncowy - czas_poczatkowy,
            'czas_poczatkowy': czas_poczatkowy
        }
        return Sygnal(dane_nowy_sygnal, all_probki, parametry, 'ciagly', 'ciagly')

    @staticmethod
    def odejmowanie(pierwszy_sygnal, drugi_sygnal):
        czas_poczatkowy = min(pierwszy_sygnal.parametry['czas_poczatkowy'], drugi_sygnal.parametry['czas_poczatkowy'])
        czas_koncowy = max(pierwszy_sygnal.parametry['czas_poczatkowy'] + pierwszy_sygnal.parametry['czas_trwania'],
                       drugi_sygnal.parametry['czas_poczatkowy'], drugi_sygnal.parametry['czas_trwania'])
        all_probki = np.arange(czas_poczatkowy, czas_koncowy, 1 / pierwszy_sygnal.parametry['czestotliwosc_probkowania'])
        dane_nowy_sygnal = []
        tmp_a = 0
        tmp_b = 0
        for probka in all_probki:
            tmp_a_index = np.where(pierwszy_sygnal.probki_czasowe == probka)[0][
                0] if probka in pierwszy_sygnal.probki_czasowe else None
            tmp_b_index = np.where(drugi_sygnal.probki_czasowe == probka)[0][
                0] if probka in drugi_sygnal.probki_czasowe else None
            tmp_a = pierwszy_sygnal.dane[tmp_a_index] if tmp_a_index is not None else 0
            tmp_b = drugi_sygnal.dane[tmp_b_index] if tmp_b_index is not None else 0
            dane_nowy_sygnal.append(tmp_a - tmp_b)
        parametry = {
            'czestotliwosc_probkowania': pierwszy_sygnal.parametry['czestotliwosc_probkowania'],
            'czas_trwania': czas_koncowy - czas_poczatkowy,
            'czas_poczatkowy': czas_poczatkowy
        }
        return Sygnal(dane_nowy_sygnal, all_probki, parametry, 'ciagly', 'ciagly')

    @staticmethod
    def dzielenie(pierwszy_sygnal, drugi_sygnal):
        czas_poczatkowy = min(pierwszy_sygnal.parametry['czas_poczatkowy'], drugi_sygnal.parametry['czas_poczatkowy'])
        czas_koncowy = max(pierwszy_sygnal.parametry['czas_poczatkowy'] + pierwszy_sygnal.parametry['czas_trwania'],
                       drugi_sygnal.parametry['czas_poczatkowy'] + drugi_sygnal.parametry['czas_trwania'])
        all_probki = np.arange(czas_poczatkowy, czas_koncowy, 1 / pierwszy_sygnal.parametry['czestotliwosc_probkowania'])
        dane_nowy_sygnal = []
        tmp_a = 0
        tmp_b = 0
        for probka in all_probki:
            tmp_a_index = np.where(pierwszy_sygnal.probki_czasowe == probka)[0][
                0] if probka in pierwszy_sygnal.probki_czasowe else None
            tmp_b_index = np.where(drugi_sygnal.probki_czasowe == probka)[0][
                0] if probka in drugi_sygnal.probki_czasowe else None
            tmp_a = pierwszy_sygnal.dane[tmp_a_index] if tmp_a_index is not None else 1
            tmp_b = drugi_sygnal.dane[tmp_b_index] if tmp_b_index is not None else 1
            dane_nowy_sygnal.append(tmp_a / tmp_b)
        parametry = {
            'czestotliwosc_probkowania': pierwszy_sygnal.parametry['czestotliwosc_probkowania'],
            'czas_trwania': czas_koncowy - czas_poczatkowy,
            'czas_poczatkowy': czas_poczatkowy
        }
        return Sygnal(dane_nowy_sygnal, all_probki, parametry, 'ciagly', 'ciagly')

class Generator:
    @staticmethod
    def szum_o_rozkladzie_jednostajnym(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania):
        dane = []
        probki = np.arange(czas_poczatkowy, czas_trwania+czas_poczatkowy, 1/czestotliwosc_probkowania)

        for probka in probki:
            dane.append(np.random.uniform() * amplituda)
        parametry = {'czestotliwosc_probkowania': czestotliwosc_probkowania, 'amplituda': amplituda, 'czas_poczatkowy': czas_poczatkowy, 'czas_trwania': czas_trwania }

        return Sygnal(dane, probki, parametry, 'szum_o_rozkladzie_jednostajnym', 'ciagly')

    @staticmethod
    def szum_gaussowski(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, mean=0, std=1):
        dane = []
        probki = np.arange(czas_poczatkowy, czas_trwania+czas_poczatkowy, 1/czestotliwosc_probkowania)

        for probka in probki:
            dane.append(np.random.normal(mean, std) * amplituda)
        parametry = {'czestotliwosc_probkowania': czestotliwosc_probkowania,'czas_poczatkowy': czas_poczatkowy,'amplituda': amplituda,'czas_trwania': czas_trwania }

        return Sygnal(dane, probki, parametry, 'szum_gaussowski', 'ciagly')

    @staticmethod
    def sygnal_sinusoidalny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania):
        dane = []
        probki = np.arange(czas_poczatkowy, czas_trwania+czas_poczatkowy, 1/czestotliwosc_probkowania)

        for probka in probki:
            dane.append(amplituda * np.sin((np.pi * 2 / okres) * (probka - czas_poczatkowy)))
        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'okres': okres,'czas_poczatkowy': czas_poczatkowy, 'czas_trwania': czas_trwania}

        return Sygnal(dane, probki, parametry, 'sygnal_sinusoidalny', 'ciagly')

    @staticmethod
    def sygnal_sinusoidalny_wyprostowany_jednopolowkowo(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania):
        dane = []
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)

        for probka in probki:
            x = 0.5 * amplituda * (np.sin((2 * np.pi / okres) * (probka - czas_poczatkowy)) +
                                   np.abs(np.sin((2 * np.pi / okres) * (probka - czas_poczatkowy))))
            dane.append(x)

        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'okres': okres,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania}

        return Sygnal(dane, probki, parametry, 'sygnal_sinusoidalny_wyprostowany_jednopolowkowo', 'ciagly')

    @staticmethod
    def sygnal_sinusoidalny_wyprostowany_dwupolowkowo(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania):
        dane = []
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)

        for probka in probki:
            dane.append(amplituda * np.abs(np.sin((2 * np.pi / okres)*(probka - czas_poczatkowy))))

        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'okres': okres,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania}

        return Sygnal(dane, probki, parametry, 'sygnal_sinusoidalny_wyprostowany_dwupolowkowo', 'ciagly')

    @staticmethod
    def sygnal_prostokatny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania, stopien_wypelnienia):
        dane = []
        x = 0
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)

        for probka in probki:
            if probka >= czas_poczatkowy + (x + 1) * okres:
                x += 1
            if probka >= (x * okres) + czas_poczatkowy and probka < (stopien_wypelnienia * okres) + (x * okres) + czas_poczatkowy:
                dane.append(amplituda)
            elif probka >= (stopien_wypelnienia * okres) - (x * okres) + czas_poczatkowy and okres + (x * okres) + czas_poczatkowy:
                dane.append(0)

        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'okres': okres,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania,'stopien_wypelnienia': stopien_wypelnienia}

        return Sygnal(dane, probki, parametry, 'sygnal_prostokatny', 'ciagly')

    @staticmethod
    def sygnal_prostokatny_symetryczny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania, stopien_wypelnienia):
        dane = []
        k = 0
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)

        for probka in probki:
            if probka >= czas_poczatkowy + (k + 1) * okres:
                k += 1
            if probka >= (k * okres) + czas_poczatkowy and probka < (stopien_wypelnienia * okres) + (k * okres) + czas_poczatkowy:
                dane.append(amplituda)
            elif probka >= (stopien_wypelnienia * okres) + czas_poczatkowy and probka < okres + (k * okres) + czas_poczatkowy:
                dane.append(-amplituda)
        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'okres': okres,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania,'stopien_wypelnienia': stopien_wypelnienia}

        return Sygnal(dane, probki, parametry, 'sygnal_prostokatny_symetryczny', 'ciagly')

    @staticmethod
    def sygnal_trojkatny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania, stopien_wypelnienia):
        dane = []
        k = 0
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)

        for probka in probki:
            if probka >= czas_poczatkowy + (k + 1) * okres:
                k += 1
            if probka >= (k * okres) + czas_poczatkowy and probka < (stopien_wypelnienia * okres) + (k * okres) + czas_poczatkowy:
                dane.append((amplituda / (stopien_wypelnienia * okres)) * (probka - (k * okres) - czas_poczatkowy))
            elif probka >=  (stopien_wypelnienia * okres) + czas_poczatkowy + (k * okres) and probka < okres + (k * okres) + czas_poczatkowy:
                dane.append((-amplituda / (okres * (1 - stopien_wypelnienia))) * (probka - (k * okres) - czas_poczatkowy) + (amplituda / (1 - stopien_wypelnienia)))

        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'okres': okres,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania,'stopien_wypelnienia': stopien_wypelnienia }

        return Sygnal(dane, probki, parametry, 'sygnal_trojkatny', 'ciagly')

    @staticmethod
    def skok_jednostkowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, punkt_skoku):
        dane = []
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)

        for probka in probki:
            if probka > punkt_skoku:
                dane.append(amplituda)
            elif probka == punkt_skoku:
                dane.append(0.5 * amplituda)
            elif probka < punkt_skoku:
                dane.append(0)
        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania,'punkt_skoku': punkt_skoku}
        return Sygnal(dane, probki, parametry, 'skok_jednostkowy', 'ciagly')

    @staticmethod
    def impuls_jednostkowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, punkt_skoku):
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)
        best_fit_probka_index = -1
        best_fit_val = 10000
        for index, probka in enumerate(probki):
            val = np.abs(probka - punkt_skoku)
            if val < best_fit_val:
                best_fit_probka_index = index
                best_fit_val = val
        dane = [amplituda if index == best_fit_probka_index else 0 for index, probka in enumerate(probki)]
        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania,'punkt_skoku': punkt_skoku }
        return Sygnal(dane, probki, parametry, 'impuls_jednostkowy', 'dyskretny')

    @staticmethod
    def szum_impulsowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, prawdopodobienstwo):
        probki = np.arange(start=czas_poczatkowy, stop=czas_trwania+czas_poczatkowy, step=1/czestotliwosc_probkowania)

        dane = [amplituda if np.random.random() < prawdopodobienstwo else 0 for probka in probki]

        parametry = {'amplituda': amplituda,'czestotliwosc_probkowania': czestotliwosc_probkowania,'czas_poczatkowy': czas_poczatkowy,'czas_trwania': czas_trwania,'prawdopodobienstwo': prawdopodobienstwo}

        return Sygnal(dane, probki, parametry, 'szum_impulsowy', 'dyskretny')

def start_gui():
    sg.theme('DarkBrown') # Wygląd GUI
    # Layout
    layout = [[
                [sg.Button('Zapisz do pliku'), sg.Button('Wczytaj z pliku'), sg.Button('Dodawanie'), sg.Button('Odejmowanie'), sg.Button('Dzielenie'), sg.Button('Mnozenie')],

                [sg.Checkbox('Drugi sygnal', enable_events=True, key='-chceck-')],
                [sg.Combo(['szum o rozkładzie jednostajnym','szum gaussowski','sygnał sinusoidalny','sygnał sinusoidalny wyprostowany jednopołówkowo',' sygnał sinusoidalny wyprostowany dwupołówkowo','sygnał prostokątny','sygnał prostokątny symetryczny','sygnał trójkątny','skok jednostkowy','impuls jednostkowy','szum impulsowy',],size=(70,1), enable_events=True, key='combo' )],
                [sg.Combo(['szum o rozkładzie jednostajnym', 'szum gaussowski', 'sygnał sinusoidalny',
                   'sygnał sinusoidalny wyprostowany jednopołówkowo', ' sygnał sinusoidalny wyprostowany dwupołówkowo',
                   'sygnał prostokątny', 'sygnał prostokątny symetryczny', 'sygnał trójkątny', 'skok jednostkowy',
                   'impuls jednostkowy', 'szum impulsowy', ], size=(70, 1), enable_events=True, key='combo2')],

                [sg.Text('sygnal pierwszy', size=(30, 1)),
                 sg.Text('sygnal drugi', size=(20, 1))],

                [sg.Text('czestotliowsc_probkowania', size=(20, 1)) , sg.In(size=(10, 1), key='-czestotliowsc_probkowania-', default_text='0'),
                 sg.Text('czestotliowsc_probkowania', size=(20, 1)) , sg.In(size=(10, 1), key='-czestotliowsc_probkowania2-', default_text='0')],

                [sg.Text('amplituda', size=(20, 1)), sg.In(size=(10,1), key='-amplituda-', default_text='0'),
                 sg.Text('amplituda', size=(20, 1)), sg.In(size=(10,1), key='-amplituda2-', default_text='0')],

                [sg.Text('okres', size=(20, 1)), sg.In(size=(10,1), key='-okres-', default_text='0'),
                 sg.Text('okres', size=(20, 1)), sg.In(size=(10,1), key='-okres2-', default_text='0')],

                [sg.Text('czas_poczatkowy', size=(20, 1)), sg.In(size=(10,1), key='-czas_poczatkowy-', default_text='0'),
                 sg.Text('czas_poczatkowy', size=(20, 1)), sg.In(size=(10,1), key='-czas_poczatkowy2-', default_text='0')],

                [sg.Text('czas_trwania', size=(20, 1)), sg.In(size=(10,1), key='-czas_trwania-', default_text='0'),
                 sg.Text('czas_trwania', size=(20, 1)), sg.In(size=(10,1), key='-czas_trwania2-', default_text='0')],

                [sg.Text('wskaźnik wypełnienia', size=(20, 1)), sg.In(size=(10,1), key='-wskaznik-', default_text='0'),
                 sg.Text('wskaźnik wypełnienia', size=(20, 1)), sg.In(size=(10,1), key='-wskaznik2-', default_text='0')],

                [sg.Text('przeskok', size=(20, 1)), sg.In(size=(10, 1), key='-przeskok-', default_text='0'),
                 sg.Text('przeskok', size=(20, 1)), sg.In(size=(10, 1), key='-przeskok2-', default_text='0')],

                [sg.Text('prawdopodobieństwo', size=(20, 1)), sg.In(size=(10, 1), key='-prawdopodobienstwo-', default_text='0'),
                 sg.Text('prawdopodobieństwo', size=(20, 1)), sg.In(size=(10, 1), key='-prawdopodobienstwo2-', default_text='0')],
                [sg.Button(('Ok'), size=(30, 1)), sg.Button(('Cancel'), size=(30, 1))]

    ]]
    # Create the Window
    window = sg.Window('Generator sygnału', layout)

    global sygnal1
    global sygnal2

    czy_drugi = False

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        if event.startswith('-chceck-'):
            czy_drugi = True

        if event == 'Ok':
            #Pobieranie wartości z GUI
            combo = values['combo']
            combo2 = values['combo2']

            czestotliwosc_probkowania = float(values['-czestotliowsc_probkowania-'])
            amplituda = float(values['-amplituda-'])
            okres = float(values['-okres-'])
            czas_poczatkowy = float(values['-czas_poczatkowy-'])
            czas_trwania = float(values['-czas_trwania-'])
            wskaznik = float(values['-wskaznik-'])
            przeskok = float(values['-przeskok-'])
            prawdopodobienstwo = float(values['-prawdopodobienstwo-'])

            czestotliwosc_probkowania2 = float(values['-czestotliowsc_probkowania2-'])
            amplituda2 = float(values['-amplituda2-'])
            okres2 = float(values['-okres2-'])
            czas_poczatkowy2 = float(values['-czas_poczatkowy2-'])
            czas_trwania2 = float(values['-czas_trwania2-'])
            wskaznik2 = float(values['-wskaznik2-'])
            przeskok2= float(values['-przeskok2-'])
            prawdopodobienstwo2 = float(values['-prawdopodobienstwo2-'])

            ############################################
            if combo == 'sygnał sinusoidalny':
                sygnal1 = Generator.sygnal_sinusoidalny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'sygnał sinusoidalny':
                sygnal2 = Generator.sygnal_sinusoidalny(czestotliwosc_probkowania2, amplituda2, okres2,
                                                            czas_poczatkowy2, czas_trwania2)
            #############################################
            if combo == 'szum o rozkładzie jednostajnym':
                sygnal1 = Generator.szum_o_rozkladzie_jednostajnym(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'szum o rozkładzie jednostajnym':
                sygnal2 = Generator.szum_o_rozkladzie_jednostajnym(czestotliwosc_probkowania, amplituda,
                                                                       czas_poczatkowy, czas_trwania)
            #############################################
            if combo == 'szum gaussowski':
                sygnal1 = Generator.szum_gaussowski(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, mean=0, std=1)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'szum gaussowski':
                sygnal2 = Generator.szum_gaussowski(czestotliwosc_probkowania, amplituda, czas_poczatkowy,
                                                        czas_trwania, mean=0, std=1)
            #############################################
            if combo == 'sygnał sinusoidalny wyprostowany jednopołówkowo':
                sygnal1 = Generator.sygnal_sinusoidalny_wyprostowany_jednopolowkowo(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'sygnał sinusoidalny wyprostowany jednopołówkowo':
                sygnal2 = Generator.sygnal_sinusoidalny_wyprostowany_jednopolowkowo(czestotliwosc_probkowania,
                                                                                        amplituda, okres,
                                                                                        czas_poczatkowy, czas_trwania)
            #############################################
            if combo == 'sygnał sinusoidalny wyprostowany dwupołówkowo':
                sygnal1 = Generator.sygnal_sinusoidalny_wyprostowany_dwupolowkowo(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'sygnał sinusoidalny wyprostowany dwupołówkowo':
                sygnal2 = Generator.sygnal_sinusoidalny_wyprostowany_dwupolowkowo(czestotliwosc_probkowania,
                                                                                      amplituda, okres, czas_poczatkowy,
                                                                                      czas_trwania)
            #############################################
            if combo == 'sygnał prostokątny':
                sygnal1 = Generator.sygnal_prostokatny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania, wskaznik)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'sygnał prostokątny':
                 sygnal2 = Generator.sygnal_prostokatny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy,
                                                           czas_trwania, wskaznik)
            #############################################
            if combo == 'sygnał prostokątny symetryczny':
                sygnal1 = Generator.sygnal_prostokatny_symetryczny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania, wskaznik)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'sygnał prostokątny symetryczny':
                sygnal2 = Generator.sygnal_prostokatny_symetryczny(czestotliwosc_probkowania, amplituda, okres,
                                                                       czas_poczatkowy, czas_trwania, wskaznik)
            #############################################
            if combo == 'sygnał trójkątny':
                sygnal1 = Generator.sygnal_trojkatny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy, czas_trwania, wskaznik)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'sygnał trójkątny':
                sygnal2 = Generator.sygnal_trojkatny(czestotliwosc_probkowania, amplituda, okres, czas_poczatkowy,
                                                         czas_trwania, wskaznik)
            #############################################
            if combo == 'skok jednostkowy':
                sygnal1 = Generator.skok_jednostkowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, przeskok)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'skok jednostkowy':
                sygnal2 = Generator.skok_jednostkowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy,
                                                         czas_trwania, przeskok)
            #############################################
            if combo == 'impuls jednostkowy':
                sygnal1 = Generator.impuls_jednostkowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, przeskok)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'impuls jednostkowy':
                sygnal2 = Generator.impuls_jednostkowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy,
                                                           czas_trwania, przeskok)
            #############################################
            if combo == 'szum impulsowy':
                sygnal1 = Generator.szum_impulsowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy, czas_trwania, prawdopodobienstwo)
                Sygnal.wynik_okno(sygnal1)
            if czy_drugi is True and combo2 == 'szum impulsowy':
                sygnal2 = Generator.szum_impulsowy(czestotliwosc_probkowania, amplituda, czas_poczatkowy,
                                                       czas_trwania, prawdopodobienstwo)

        if event == 'Zapisz do pliku':
            if sygnal1 is None:
                print("Blad")
            else:
                Sygnal.zapisz_do_pliku(sygnal1, 'sygnal')

        if event == 'Wczytaj z pliku':
            wczytane = Sygnal.wczytaj_z_pliku('sygnal')
            Sygnal.wykresy(wczytane)

        if event == 'Dodawanie':
            wynik1 = Sygnal.dodawanie(sygnal1, sygnal2)
            Sygnal.wynik_okno(wynik1)

        if event == 'Odejmowanie':
            wynik2 = Sygnal.odejmowanie(sygnal1, sygnal2)
            Sygnal.wynik_okno(wynik2)

        if event == 'Dzielenie':
            wynik3 = Sygnal.dzielenie(sygnal1, sygnal2)
            Sygnal.wynik_okno(wynik3)

        if event == 'Mnozenie':
            wynik4 = Sygnal.mnozenie(sygnal1, sygnal2)
            Sygnal.wynik_okno(wynik4)


    window.close()

start_gui()