from imports import *
from util_func import *

if __name__ == "__main__":

    subject, data_path, meta_path, condition = start_session()

    if condition["condition"] == "relearn":
        ds = make_stim_cats()
        ds["condition"] = "relearn"

    else:
        ds = make_stim_cats()
        ds["condition"] = "new_learn"
        ds_train = ds.iloc[:600].copy()
        ds_test = ds.iloc[600:].copy()
        ds_test["cat"] = ds_test["cat"].map({
            "A": "B",
            "B": "C",
            "C": "D",
            "D": "A"
        })
        ds_test["cat"] = pd.Categorical(ds_test["cat"],
                                        categories=["A", "B", "C", "D"],
                                        ordered=True)
        ds = pd.concat([ds_train, ds_test]).reset_index(drop=True)

    # Optional debug plot of generated stimuli (disabled for normal runtime):
    # import seaborn as sns
    # fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(12, 6))
    # fig.suptitle(
    #     f"Experiment {condition['experiment']} - {condition['condition']}")
    # sns.scatterplot(data=ds.iloc[:600],
    #                 x="x",
    #                 y="y",
    #                 hue="cat",
    #                 alpha=0.5,
    #                 ax=ax[0, 0])
    # sns.scatterplot(data=ds.iloc[:600],
    #                 x="xt",
    #                 y="yt",
    #                 hue="cat",
    #                 alpha=0.5,
    #                 ax=ax[0, 1])
    # sns.scatterplot(data=ds.iloc[600:],
    #                 x="x",
    #                 y="y",
    #                 hue="cat",
    #                 alpha=0.5,
    #                 ax=ax[1, 0])
    # sns.scatterplot(data=ds.iloc[600:],
    #                 x="xt",
    #                 y="yt",
    #                 hue="cat",
    #                 alpha=0.5,
    #                 ax=ax[1, 1])
    # ax[0, 0].plot([0, 100], [0, 100], 'k--')
    # ax[1, 0].plot([0, 100], [0, 100], 'k--')
    # ax[0, 0].plot([0, 100], [100, 0], 'k--')
    # ax[1, 0].plot([0, 100], [100, 0], 'k--')
    # ax[0, 1].plot([0, 5], [0, np.pi / 2], 'k--')
    # ax[1, 1].plot([0, 5], [0, np.pi / 2], 'k--')
    # ax[1, 1].plot([0, 5], [0, np.pi / 2], 'k--')
    # ax[1, 1].plot([0, 5], [np.pi / 2, 0], 'k--')
    # [ax[i, j].legend(loc='upper right') for i in range(2) for j in range(2)]
    # plt.show()

    # plot_stim_space_examples(ds)

    # Initialize Pygame
    pygame.init()

    # useful constants but need to change / verify on each computer
    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54

    # grating size
    size_cm = 5
    size_px = int(size_cm * px_per_cm)

    # set small window potentially useful for debugging
    # screen_width, screen_height = 800, 600
    # center_x = screen_width // 2
    # center_y = screen_height // 2
    # screen = pygame.display.set_mode((screen_width, screen_height))

    # set full screen
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    center_x = screen_width // 2
    center_y = screen_height // 2
    screen = pygame.display.set_mode((screen_width, screen_height),
                                     pygame.FULLSCREEN)

    # Hide the mouse cursor
    pygame.mouse.set_visible(False)

    # Set up fonts
    font = pygame.font.Font(None, 36)

    # Define colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    grey = (126, 126, 126)
    green = (0, 255, 0)
    red = (255, 0, 0)

    # create clocks to keep time
    clock_state = pygame.time.Clock()
    clock_exp = pygame.time.Clock()

    time_state = 0.0
    time_exp = 0.0

    # set the current state to the initial state
    state_current = "state_init"

    # behavioural measurements
    resp = -1
    rt = -1

    # trial counter
    trial = -1
    n_trial = ds.shape[0]

    # record keeping
    trial_data = {
        'experiment': [],
        'condition': [],
        'subject': [],
        'trial': [],
        'cat': [],
        'x': [],
        'y': [],
        'xt': [],
        'yt': [],
        'resp': [],
        'rt': [],
        'fb': []
    }

    running = True
    while running:

        time_exp += clock_exp.tick()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                    break
                else:
                    resp = event.key

        if not running:
            break

        if state_current == "state_explicit_instruct":
            time_state += clock_state.tick()

            message_instruct_1 = "Over the last many trials the feedback you received was random."
            message_instruct_2 = "This was an important part of the experiment."
            message_instruct_3 = "From now on, the feedback will again be valid."
            message_instruct_4 = "Please keep trying to categorize correctly."
            message_instruct_5 = "Press the Y key to proceed"
            message_instruct = [
                message_instruct_1, message_instruct_2, message_instruct_3,
                message_instruct_4, message_instruct_5
            ]

            spacing = screen_height / 8
            screen.fill(grey)
            for ii, mm in enumerate(message_instruct):
                text = font.render(mm, True, (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width / 2,
                                                  screen_height / 4 +
                                                  ii * spacing))
                screen.blit(text, text_rect)

            if resp == pygame.K_y:
                time_state = 0
                resp = -1
                state_current = "state_iti"

        if state_current == "state_init":
            time_state += clock_state.tick()
            text = font.render("Please press the space bar to begin", True,
                               (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2,
                                              screen_height / 2))
            screen.fill(grey)
            screen.blit(text, text_rect)
            if resp == pygame.K_SPACE:
                time_state = 0
                resp = -1
                state_current = "state_iti"

        if state_current == "state_finished":
            time_state += clock_state.tick()
            text = font.render("You finished! Thank you for being awesome!",
                               True, (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width / 2,
                                              screen_height / 2))
            screen.fill(grey)
            screen.blit(text, text_rect)

        if state_current == "state_iti":
            time_state += clock_state.tick()
            screen.fill(grey)
            pygame.draw.line(screen, white, (center_x, center_y - 10),
                             (center_x, center_y + 10), 4)
            pygame.draw.line(screen, white, (center_x - 10, center_y),
                             (center_x + 10, center_y), 4)
            if time_state > 1000:
                resp = -1
                rt = -1
                time_state = 0
                trial += 1
                if trial == n_trial - 1:
                    state_current = "state_finished"
                else:
                    sf = ds['xt'].iloc[trial] * (px_per_cm**-1)
                    ori = ds['yt'].iloc[trial]
                    cat = ds['cat'].iloc[trial]
                    resp = -1
                    state_current = "state_stim"

        if state_current == "state_stim":
            time_state += clock_state.tick()
            screen.fill(grey)
            grating_patch = create_grating_patch(size_px, sf, ori)
            grating_surface = grating_to_surface(grating_patch)
            screen.blit(grating_surface,
                        (center_x - size_px / 2, center_y - size_px / 2))

            listen_keys = [pygame.K_d, pygame.K_f, pygame.K_j, pygame.K_k]
            if resp in listen_keys:
                rt = time_state
                time_state = 0

                if resp == pygame.K_d:
                    resp = "A"
                elif resp == pygame.K_f:
                    resp = "B"
                elif resp == pygame.K_j:
                    resp = "C"
                elif resp == pygame.K_k:
                    resp = "D"

                # Give veridical feedback in all conditions during acquisition and reacquisition
                if (trial < 300) or (trial >= 600):
                    if cat == resp:
                        fb = "Correct"
                    else:
                        fb = "Incorrect"

                else:
                    # Exp 1: random feedback during intervention
                    if condition["experiment"] == 1:
                        # random feedback
                        if np.random.rand() < 0.5:
                            fb = "Correct"
                        else:
                            fb = "Incorrect"

                    # Exp 2: 7525 mixed intervention
                    else:
                        # give veridical feedback 25% of the time
                        if np.random.rand() < 0.25:
                            if cat == resp:
                                fb = "Correct"
                            else:
                                fb = "Incorrect"

                        # Give 100% random feedback the rest of the time
                        else:
                            if np.random.rand() < 0.5:
                                fb = "Correct"
                            else:
                                fb = "Incorrect"

                state_current = "state_feedback"

        if state_current == "state_feedback":
            time_state += clock_state.tick()

            if fb == "Correct":
                pygame.draw.circle(screen, green, (center_x, center_y),
                                   size_px / 2 + 10, 5)

            elif fb == "Incorrect":
                pygame.draw.circle(screen, red, (center_x, center_y),
                                   size_px / 2 + 10, 5)

            if time_state > 1000:
                trial_data['experiment'].append(condition["experiment"])
                trial_data['condition'].append(condition["condition"])
                trial_data['subject'].append(subject)
                trial_data['trial'].append(trial)
                trial_data['cat'].append(cat)
                trial_data['x'].append(ds.x[trial])
                trial_data['y'].append(ds.y[trial])
                trial_data['xt'].append(ds.xt[trial])
                trial_data['yt'].append(ds.yt[trial])
                trial_data['resp'].append(resp)
                trial_data['rt'].append(rt)
                trial_data['fb'].append(fb)
                pd.DataFrame(trial_data).to_csv(data_path, index=False)
                time_state = 0

                # transition to test gets explicit instructions
                if trial == 600:
                    state_current = "state_explicit_instruct"
                else:
                    state_current = "state_iti"

        pygame.display.flip()
