import numpy as np

class Moment:
    """A class for keeping info about the moments"""
    def __init__(self, x_true, x_pred,t,index_all,x_static,args,player_ids, player_names, player_jerseys,team_O,team_D,quarter,CF=False): 
        n_playersxy = args.x_dim_permuted-2 if not args.vel else args.x_dim_permuted-4 # 4
        x_dim = 2 if not args.vel else 4 # 4
        self.quarter = int(quarter)
        player_id = x_static[1:11,np.newaxis]
        team_Os = np.repeat(team_O[:,np.newaxis],5,0)
        team_Ds = np.repeat(team_D[:,np.newaxis],5,0)
        player_team = np.concatenate([team_Os,team_Ds],0)
        GP_prevs = ['' for _ in range(10)]

        if CF and t >= args.burn_in:
            self.game_clock = x_pred[t,n_playersxy+9]*720
            self.shot_clock = x_pred[t,n_playersxy+10]*24
            GP_prev = x_pred[t,n_playersxy+7]
            
            ball = np.concatenate([x_pred[t,n_playersxy:n_playersxy+2],x_pred[t,-1:]],0)
            player_xy = x_pred[t,:n_playersxy].reshape((10,x_dim))[:,:2]    
            # ball = np.concatenate([x_true[t,n_playersxy:n_playersxy+2],x_true[t,-1:]],0)
            # player_xy = x_pred[t,:n_playersxy].reshape((10,x_dim))[:,:2]            
            Ball_OF_dist = np.sum((player_xy[:5,:]-np.repeat(ball[np.newaxis,:2],5,0))**2,1)
            ball_player = np.argmin(Ball_OF_dist)
        else:
            self.game_clock = x_true[t,n_playersxy+9]*720
            self.shot_clock = x_true[t,n_playersxy+10]*24
            ball = np.concatenate([x_true[t,n_playersxy:n_playersxy+2],x_true[t,-1:]],0)
            
            self.ball = self.Ball(ball)
            
            player_xy = x_true[t,:n_playersxy].reshape((10,x_dim))[:,:2]          
            Ball_OF_dist = np.sum((player_xy[:5,:]-np.repeat(ball[np.newaxis,:2],5,0))**2,1)
            ball_player = np.argmin(Ball_OF_dist)
            
            GP_prev = x_true[t,n_playersxy+7]

            players = np.concatenate([player_id,player_xy,player_team],1)
            self.players = [self.Player(player,GP_prev) for player,GP_prev in zip(players, GP_prevs)]
            self.team = [self.Team(int(player[3]),args) for player in players]
            self.color = [self.team[p].color for p in range(10)]
        # pred
        
        ball_pred = np.concatenate([x_pred[t,n_playersxy:n_playersxy+2],x_pred[t,-1:]],0)
        try: self.ball_pred = self.Ball(ball_pred,pred=True)
        except: import pdb; pdb.set_trace()
        player_xy_pred = x_pred[t,:n_playersxy].reshape((10,x_dim))[:,:2]
        
        players_pred = np.concatenate([player_id,player_xy_pred,player_team],1)
        self.players_pred = [self.Player(player,GP_prev) for player,GP_prev in zip(players_pred, GP_prevs)]
        self.team_pred = [self.Team(int(player[3]),args,pred=True) for player in players_pred]
        self.color_pred = [self.team_pred[p].color for p in range(10)]

        # players info
        player_id = x_static[1:11].astype(np.int64)
        id_index = [np.where(player_ids==id)[0] for id in player_id]
        try: player_name = [str(player_names[id][0]) for id in id_index]
        except: import pdb; pdb.set_trace()
        player_jersey = [int(player_jerseys[id]) for id in id_index]
        values = list(zip(player_name, player_jersey,GP_prevs))
        # Example: 101108: ['Chris Paul', '3']
        self.player_ids_dict = dict(zip(player_id, values))

        GP_prevs[ball_player] = '('+str(GP_prev)+')'
        self.GP_prev = '#'+str(player_jersey[ball_player])+' {:.3f}'.format(GP_prev)

    class Ball:
        """A class for keeping info about the balls"""
        def __init__(self, ball,pred=False):
            self.x = ball[0]
            self.y = ball[1]
            
            if pred:
                self.radius = np.array([2])
                self.color = 'white'
            else:
                self.radius = ball[2]
                self.color = '#ff8c00'  # Hardcoded orange
    class Player:
        """A class for keeping info about the players"""
        def __init__(self, player, GP_prev):
            self.id = player[0]
            self.x = player[1]
            self.y = player[2]
            self.GP_prev = GP_prev

    class Team:
        """A class for keeping info about the teams"""
        def __init__(self, id, args, pred=False):
            color_dict = {
                1610612737: ('#E13A3E', 'ATL'),
                1610612738: ('#008348', 'BOS'),
                1610612751: ('#061922', 'BKN'),
                1610612766: ('#1D1160', 'CHA'),
                1610612741: ('#CE1141', 'CHI'),
                1610612739: ('#860038', 'CLE'),
                1610612742: ('#007DC5', 'DAL'),
                1610612743: ('#4D90CD', 'DEN'),
                1610612765: ('#006BB6', 'DET'),
                1610612744: ('#FDB927', 'GSW'),
                1610612745: ('#CE1141', 'HOU'),
                1610612754: ('#00275D', 'IND'),
                1610612746: ('#ED174C', 'LAC'),
                1610612747: ('#552582', 'LAL'),
                1610612763: ('#0F586C', 'MEM'),
                1610612748: ('#98002E', 'MIA'),
                1610612749: ('#00471B', 'MIL'),
                1610612750: ('#005083', 'MIN'),
                1610612740: ('#002B5C', 'NOP'),
                1610612752: ('#006BB6', 'NYK'),
                1610612760: ('#007DC3', 'OKC'),
                1610612753: ('#007DC5', 'ORL'),
                1610612755: ('#006BB6', 'PHI'),
                1610612756: ('#1D1160', 'PHX'),
                1610612757: ('#E03A3E', 'POR'),
                1610612758: ('#724C9F', 'SAC'),
                1610612759: ('#BAC3C9', 'SAS'),
                1610612761: ('#CE1141', 'TOR'),
                1610612762: ('#00471B', 'UTA'),
                1610612764: ('#002B5C', 'WAS'),
            }

            self.id = id
            if pred and args.movie:
                self.color = 'white'
            else:
                self.color = color_dict[id][0]
            self.name = color_dict[id][1]

class Constant:
    """A class for handling constants"""
    Half = True
    feet_m = 0.3048

    NORMALIZATION_COEF = 3 # 7
    PLAYER_CIRCLE_SIZE = 12 / 7 *feet_m
    INTERVAL = 10
    DIFF = 6*feet_m
    X_MIN = 0
    X_MAX = 100*feet_m/2 if Half else 100*feet_m
    Y_MIN = 0
    Y_MAX = 50*feet_m
    COL_WIDTH = 0.45 if Half else 0.3
    SCALE = 1.65
    # FONTSIZE = 6
    X_CENTER = X_MAX / 2 - DIFF/2 / 1.5 + 0.10*feet_m if Half else X_MAX / 2 - DIFF / 1.5 + 0.10*feet_m
    Y_CENTER = Y_MAX - DIFF / 1.5 - 0.35*feet_m
    MESSAGE = 'You can rerun the script and choose any event from 0 to '


