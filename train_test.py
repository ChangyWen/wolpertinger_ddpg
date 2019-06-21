#!/usr/bin/env python
# -*- coding: utf-8 -*-

def train(env, agent, max_episode, warmup, save_model_dir, max_episode_length, logger):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None
    while episode < max_episode:
        while True:
            if s_t is None:
                s_t = env.reset()
                agent.reset(s_t)

            # agent pick action ...
            # args.warmup: time without training but only filling the memory
            if step <= warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(s_t)

            # env response with next_observation, reward, terminate_info
            s_t1, r_t, done, _ = env.step(action)

            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True

            # agent observe and update policy
            agent.observe(r_t, s_t1, done)
            if step > warmup:
                agent.update_policy()

            # update
            step += 1
            episode_steps += 1
            episode_reward += r_t
            s_t = s_t1
            # s_t = deepcopy(s_t1)

            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(episode, episode_reward)
                )

                agent.memory.append(
                    s_t,
                    agent.select_action(s_t),
                    0., True
                )

                # reset
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                episode += 1
                # break to next episode
                break
        # [optional] save intermideate model every run through of 32 episodes
        if step > warmup and episode > 0 and episode % 2 == 0:
            agent.save_model(save_model_dir)
            logger.info("### Model Saved before Ep:{0} ###".format(episode))

# def test(agent, test_month, model_path, seed, logger):
#
#     agent.load_weights(model_path)
#     agent.is_training = False
#     agent.eval()
#
#     policy = lambda x: agent.select_action(x, decay_epsilon=False)
#
#     for m in range(test_month):
#         env = RideSharing_Env(m, seed)
#         month_reward = 0.
#         month_unfilled = month_total_requests = month_answer = month_real_total = 0
#         for day in range(31):
#             episode_reward = 0.
#             episode_unfilled = episode_total_requests = episode_answer = episode_real_total = 0
#             s_t, _ = env.reset(day)
#             done = False
#             while not done:
#                 action = policy(s_t)
#                 s_t, r_t, done, _, unfilled, total_requests, _ = env.step(action)
#                 episode_reward += r_t
#                 episode_unfilled += unfilled
#                 episode_total_requests += total_requests
#                 episode_answer += sum(env.ans_dict.values())
#                 episode_real_total += len(env.ans_dict)
#             logger.info(
#                 "Month:{0} | Day:{1} | R:{2:.4f} | UR:{3:.4f} | AR:{4:.4f}".
#                     format(m,
#                            day,
#                            episode_reward,
#                            episode_unfilled / episode_total_requests,
#                            episode_answer / episode_real_total,
#                     )
#             )
#             month_reward += episode_reward
#             month_unfilled += episode_unfilled
#             month_total_requests += episode_total_requests
#             month_answer += episode_answer
#             month_real_total += episode_real_total
#         logger.info(
#             "Month:{0} | T_R:{1:.4f} | T_UR:{2:.4f} | T_AR:{3:.4f}".
#                 format(m,
#                        month_reward,
#                        month_unfilled / month_total_requests,
#                        month_answer / month_real_total,
#                 )
#         )
