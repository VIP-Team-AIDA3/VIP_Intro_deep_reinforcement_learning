<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" defer></script>

## Tutorial 6 - Q-Learning

### Overview

Tutorial 5 introduced Temporal-Difference (TD) learning and its first control method, **SARSA**. **Q-Learning** is TD control's other major variant, and it is best understood not as a new idea but as a single, deliberate change to the SARSA update.

**What stays the same:** Q-Learning is model-free, estimates action values $Q(s,a)$ directly, updates after every single time step using a TD error rather than waiting for a full episode return, fits the same Generalized Policy Iteration (GPI) pattern as SARSA, and still requires an exploratory behavior policy (e.g. $\varepsilon$-greedy) to visit state-action pairs it hasn't tried yet.

**What changes:** SARSA bootstraps its target using $Q(S_{t+1}, A_{t+1})$, where $A_{t+1}$ is the action the behavior policy *actually selects* next. Q-Learning instead bootstraps using $\max_a Q(S_{t+1}, a)$ — the value of the *best* action available in $S_{t+1}$, regardless of which action the behavior policy actually goes on to take. That single substitution is the entire difference between an **on-policy** method (SARSA, which learns the value of the policy it is following, exploration and all) and an **off-policy** method (Q-Learning, which learns the value of the greedy policy no matter how it is currently behaving).

**A common point of confusion:** if Q-Learning's target always uses the greedy $\max_a$ action, what is the exploratory ($\varepsilon$-greedy) behavior policy actually *for*? The two policies are doing two different jobs around a *single* $Q$-table, not maintaining separate value functions. The behavior policy only decides *which* $(S_t, A_t)$ pair gets experienced and therefore updated at all — it supplies coverage, so that actions which currently look bad still eventually get tried. It has no say in what value that update moves toward. The $\max_a Q(S_{t+1}, a)$ term is purely hypothetical: it credits landing in $S_{t+1}$ as if the *best* action will be taken from there on, even though the agent's next real action (chosen again by the exploratory behavior policy) might in fact be something else entirely. That is precisely why Q-Learning converges to $q_*(s,a)$ — "the return from taking action $a$ in state $s$, then playing optimally forever after" — regardless of how exploratory the behavior generating the data was.

**A note on notation:** earlier tutorials write $v_\pi$ and $q_\pi$ for the *true* value of a fixed policy $\pi$, reserving the subscript to mean "the return if we follow $\pi$ thereafter." Because Q-Learning's target is never the value of the policy currently generating behavior, this tutorial's $Q$ is never written as $q_\pi$ — its estimation target is the policy-independent $q_*$, so it's written using $Q$ or $q_*$ only. The contrast is clearest by placing the two Bellman equations side by side. SARSA samples the Bellman equation for $q_\pi$ (Tutorial 5), whose expectation over the next action runs over $\pi$:

$$q_\pi(s,a) = \mathbb{E}_\pi\left[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \mid S_t=s,\ A_t=a\right]$$

Q-Learning instead samples the **Bellman optimality equation** for $q_*$, which has no $\pi$ in it at all — the $\max_{a'}$ replaces the $\pi$-weighted average:

$$q_*(s,a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t=s,\ A_t=a\right]$$

---

### Q-Learning: Off-Policy TD Control

The Q-Learning update rule is:

$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

Compare this directly to SARSA's update from Tutorial 5:

$$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

The two are identical apart from the bootstrap term. This distinguishes the **target policy** — the policy whose value is being learned — from the **behavior policy** — the policy actually generating actions during training:

* In Q-Learning, the target policy is always the *greedy* policy, $\pi(s) = \arg\max_a Q(s,a)$, regardless of what the behavior policy does. The behavior policy (typically $\varepsilon$-greedy, to keep exploring) can differ arbitrarily from this greedy target policy, which is exactly what makes Q-Learning off-policy.
* In SARSA, the target policy *is* the behavior policy — there is only one policy, and it is evaluated as-is, exploratory actions included.

Because Q-Learning already bootstraps directly off $\max_a Q(S',a)$, it approximates $q_*$ — the optimal action-value function — independently of the exploration policy being followed, as long as every state-action pair continues to be visited (the same coverage requirement seen for off-policy MC in Tutorial 4). Notably, this happens *without* the importance-sampling correction that off-policy Monte Carlo prediction required: because the target only ever looks at the single best next action rather than reweighting an entire sampled trajectory, no correction for the mismatch between the two policies is needed.

#### Q-Learning Algorithm

> **Algorithm: Q-learning (off-policy TD control) for estimating $\pi \approx \pi_*$**
> 1. Initialize $Q(s, a) \in \mathbb{R}$ arbitrarily for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$, except $Q(\text{terminal}, \cdot) = 0$.
> 2. **Loop forever (for each episode):**
>    * Initialize $S$
>    * **Loop for each step of episode:**
>      * Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
>      * Take action $A$, observe $R, S'$
>      * $Q(S, A) \gets Q(S, A) + \alpha \left[ R + \gamma \max_a Q(S', a) - Q(S, A) \right]$
>      * $S \gets S'$
>    * **until** $S$ is terminal
>
> *(Sutton & Barto, 2018)*

Notice the structural difference from SARSA's pseudocode: Q-Learning chooses $A$ *inside* the step loop rather than carrying it over from the previous iteration, and there is no need to pre-select $A'$ before computing the update — the target uses $\max_a Q(S', a)$ directly, an action that need never actually be taken.

---

### SARSA vs. Q-Learning at a Glance

| Feature | SARSA | Q-Learning |
| :--- | :--- | :--- |
| **Bootstrap term** | $Q(S_{t+1}, A_{t+1})$ | $\max_a Q(S_{t+1}, a)$ |
| **On-policy or off-policy?** | On-policy | Off-policy |
| **Target policy** | Same as behavior policy | Greedy policy ($\arg\max_a Q$) |
| **Behavior policy** | $\varepsilon$-greedy (or similar) | Can be any policy with adequate coverage, e.g. $\varepsilon$-greedy |
| **Needs importance sampling?** | No | No (bootstrap avoids the mismatch) |
| **Requires $\varepsilon \to 0$ (GLIE) to converge to $\pi_*$?** | Yes | No — converges to $q_*$ even under a fixed $\varepsilon$, provided all pairs are visited infinitely often |
| **Accounts for exploration risk in the learned values?** | Yes | No |

---

### Why It Matters: The Cliff Walking Example

Sutton & Barto's classic **Cliff Walking** gridworld makes the practical consequence of on- vs. off-policy learning concrete. The agent must walk from a start state to a goal state along the edge of a cliff; stepping off the cliff yields a large negative reward and sends the agent back to the start, while every other step incurs a small negative reward. Both agents train using $\varepsilon$-greedy exploration.

* **Q-Learning** learns the value of the *greedy* policy — the optimal path directly along the cliff edge — since its updates always bootstrap off $\max_a Q(S', a)$, independent of the exploratory steps actually taken. But because it is still *behaving* $\varepsilon$-greedily during training, it occasionally steps off the cliff it has learned to walk right beside, leading to worse online performance during learning.
* **SARSA** learns the value of the policy it is *actually following*, exploratory missteps included. Because an $\varepsilon$-greedy policy occasionally strays and falls off the cliff, SARSA learns that hugging the edge is risky, and instead converges to a longer but safer path further from the cliff.

Neither result is a bug: each algorithm is correctly learning the value of the policy it was designed to evaluate. The example simply illustrates concretely what "on-policy" vs. "off-policy" costs and buys you — SARSA's values account for the exploration the agent will actually do; Q-Learning's values describe a purely greedy policy the agent isn't actually always following.

---

### Quiz

#### Question 1
In state $S'$, $Q(S', a_1) = 1.0$, $Q(S', a_2) = 3.0$, $Q(S', a_3) = 2.0$. The agent's $\varepsilon$-greedy behavior policy happens to select $a_3$ as the next action. Given $Q(S,A) = 2.0$, $R = 0$, $\gamma = 1$, $\alpha = 0.5$, compute the updated $Q(S,A)$ under (a) Q-Learning and (b) SARSA.

***Answer:***
(a) Q-Learning bootstraps off the max regardless of what was actually chosen next:
$$Q(S,A) \gets 2.0 + 0.5\left[0 + (1)\max(1.0, 3.0, 2.0) - 2.0\right] = 2.0 + 0.5(1.0) = 2.5$$
(b) SARSA bootstraps off the action actually taken, $a_3$:
$$Q(S,A) \gets 2.0 + 0.5\left[0 + (1)(2.0) - 2.0\right] = 2.0 + 0.5(0) = 2.0$$

---

#### Question 2
The Q-Learning algorithm still selects actions using an $\varepsilon$-greedy policy during training. Why does this not make Q-Learning on-policy?

***Answer:*** Whether a method is on- or off-policy depends on which policy is being *evaluated* (the target of the update), not which policy generates behavior. Q-Learning's update always targets $\max_a Q(S',a)$ — the value of the strictly greedy policy — no matter which action the $\varepsilon$-greedy behavior policy actually selects next. Since the evaluated (target) policy and the behavior policy differ, Q-Learning is off-policy.

---

#### Question 3
Write the Q-Learning update rule and identify its target policy and its behavior policy.

***Answer:*** $$Q(S_t, A_t) \gets Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$ The target policy is the greedy policy $\pi(s) = \arg\max_a Q(s,a)$, implicit in the $\max_a$ term. The behavior policy is whatever exploratory policy (commonly $\varepsilon$-greedy) is actually selecting $A_t$ during training.

---

#### Question 4
In the Cliff Walking example, why does SARSA converge to a safer, longer path while Q-Learning converges to the optimal, cliff-hugging path?

***Answer:*** SARSA's update bootstraps off the action the $\varepsilon$-greedy behavior policy actually takes next, so its learned values incorporate the real risk of occasionally stepping off the cliff during exploration — making the cliff-edge path look worse and a safer detour look better. Q-Learning's update always bootstraps off $\max_a Q(S',a)$, the best possible next action, so it learns the value of the purely greedy (optimal) policy independent of the exploratory missteps it might actually make while training.

---

#### Question 5
Does SARSA require $\varepsilon$ to decay toward zero (GLIE) to converge to the optimal policy $\pi_*$? Does Q-Learning?

***Answer:*** SARSA requires $\varepsilon \to 0$ (GLIE) to converge to $\pi_*$, since it only ever learns the value of whatever policy it is currently following — if $\varepsilon$ stays fixed, it converges to the optimal *$\varepsilon$-greedy* policy, not $\pi_*$ itself. Q-Learning does not need $\varepsilon \to 0$: its target already bootstraps off the greedy action at every update, so $Q$ converges to $q_*$ (and the derived greedy policy converges to $\pi_*$) even under a fixed $\varepsilon$, as long as every state-action pair is visited infinitely often.

---

### Sources

* **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 6: Temporal-Difference Learning.
* **Morales, M.** (2020). *Grokking Deep Reinforcement Learning*. Manning Publications.
