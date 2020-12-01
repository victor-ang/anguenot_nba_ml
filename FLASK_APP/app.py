# imports
import flask
import pickle
import numpy as np
# We use pickle to load the pre-trained model.
with open(f'model/nba_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')

# We also load a model to normalize the data
with open(f'model/nba_classifier2.pkl', 'rb') as f:
    model2 = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':  # GET Mode
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':  # POST Mode
        gp = flask.request.form['gp']
        mi = flask.request.form['mi']
        pts = flask.request.form['pts']
        fgm = flask.request.form['fgm']
        fga = flask.request.form['fga']
        fg = flask.request.form['fg']
        pm = flask.request.form['pm']
        pa = flask.request.form['pa']
        p = flask.request.form['p']
        ftm = flask.request.form['ftm']
        fta = flask.request.form['fta']
        ft = flask.request.form['ft']
        oreb = flask.request.form['oreb']
        dreb = flask.request.form['dreb']
        reb = flask.request.form['reb']
        ast = flask.request.form['ast']
        stl = flask.request.form['stl']
        blk = flask.request.form['blk']
        tov = flask.request.form['tov']
        # the form filled by the user is put in an array to be used in the model
        input_variables = np.array([gp, mi, pts, fgm, fga, fg, pm, pa, p,
                                    ftm, fta, ft, oreb, dreb, reb, ast, stl, blk, tov], dtype=float)
        input_variables = input_variables.reshape(1, -1)
        # normalization of the values of the form
        input_variables = model2.transform(input_variables)
        tmp = model.predict(input_variables)[0]
        if (tmp == 1):
            prediction = "This player will last more than 5 years in the NBA."
        else:
            prediction = "This player will NOT last more than 5 years in the NBA."
        return flask.render_template('main.html',
                                     original_input={'Games played': gp,
                                                     'Minutes played': mi,
                                                     'Points per game': pts,
                                                     'Field goals made': fgm,
                                                     'Field goals attempts': fga,
                                                     'Field goals %': fg,
                                                     '3 point made': pm,
                                                     '3 point attempts': pa,
                                                     '3 point %': p,
                                                     'Free throw made': ftm,
                                                     'Free throw attempts': fta,
                                                     'Free throw %': ft,
                                                     'Offensive rebounds': oreb,
                                                     'Offensive rebounds': dreb,
                                                     'Rebounds': reb,
                                                     'Assists': ast,
                                                     'Steals': stl,
                                                     'Blocks': blk,
                                                     'Turnovers': tov},
                                     result=prediction,
                                     )


if __name__ == '__main__':
    app.debug = False  # True
    app.run()
