<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" defer></script>

## Tutorial 4 - Monte Carlo Methods

### Overview

Unlike Dynamic Programming (DP), which requires full knowledge of environment transition dynamics $p(s', r \mid s, a)$, **Monte Carlo (MC) methods** learn value functions and optimal policies directly from sample sequences of states, actions, and rewards.

MC methods apply exclusively to **episodic tasks**: tasks that naturally break into finite episodes terminating at a terminal state. Returns $G_t$ are evaluated only after an episode terminates, making MC methods incremental on an **episode-by-episode** basis rather than step-by-step.

---

### Monte Carlo Prediction

In the prediction problem, the goal is to estimate the state-value function $v_\pi(s)$ for a fixed target policy $\pi$.

The theoretical state value is defined as the expected discounted return:

$$v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

Because the expected value is unknown, MC methods approximate it by averaging sample returns observed after visits to state $s$. By the Law of Large Numbers, as the number of observed returns approaches infinity, the sample average converges to $v_\pi(s)$.

#### First-Visit vs. Every-Visit MC

Each time state $s$ appears within an episode, it is called a **visit** to $s$.

* **First-Visit MC:** Averages the returns following only the *first* time state $s$ is encountered in an episode.
* **Every-Visit MC:** Averages the returns following *every* visit to state $s$ across all episodes.

While both methods converge to $v_\pi(s)$, First-Visit MC is widely studied due to its simple, unbiased estimate in tabular cases.

#### First-Visit MC Prediction Algorithm

> **Algorithm: First-Visit MC for Estimating $v_\pi$**
> 1. Initialize $V(s) \in \mathbb{R}$ arbitrarily for all $s \in \mathcal{S}$.
> 2. Initialize $\text{Returns}(s) \gets \text{empty list}$ for all $s \in \mathcal{S}$.
> 3. **Loop forever (for each episode):**
>    * Generate an episode following policy $\pi$: $S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T$
>    * $G \gets 0$
>    * **Loop for each step $t = T-1, T-2, \dots, 0$:**
>      * $G \gets \gamma G + R_{t+1}$
>      * **If** $S_t$ does not appear in $S_0, S_1, \dots, S_{t-1}$ (**First-Visit check**):
>        * Append $G$ to $\text{Returns}(S_t)$
>        * $V(S_t) \gets \text{average}(\text{Returns}(S_t))$

---

### Monte Carlo Estimation of Action Values

When a complete model of environment dynamics is available, state values $v(s)$ suffice to construct a policy via one-step lookahead:

$$\pi(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V(s') \right]$$

Without a model ($p(s', r \mid s, a)$ is unknown), knowing $v(s)$ is insufficient to choose actions because the agent cannot predict next states or immediate rewards. Thus, model-free MC requires estimating **action values** $q_\pi(s, a)$ directly:

$$q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

The estimation machinery mirrors state prediction, replacing visits to state $s$ with visits to state-action pair $(s, a)$.

#### The Maintaining Exploration Problem
If policy $\pi$ is deterministic, the agent will repeatedly select a single action in state $s$, leaving all other $(s, a)$ pairs unvisited. Without estimates for unvisited actions, the agent cannot determine if those actions are superior. Continual exploration across all state-action pairs must be guaranteed.

---

### Monte Carlo Control

To evaluate and improve control policies without a model, MC uses **Generalized Policy Iteration (GPI)**.

$$\pi_0 \xrightarrow{\text{evaluation}} q_{\pi_0} \xrightarrow{\text{improvement}} \pi_1 \xrightarrow{\text{evaluation}} q_{\pi_1} \dots \xrightarrow{\text{improvement}} \pi_* \xrightarrow{\text{evaluation}} q_*$$

#### Exploring Starts
To satisfy the requirement that all $(s, a)$ pairs are visited infinitely often, **Monte Carlo Exploring Starts (MCES)** assumes every episode starts with a state-action pair sampled randomly with non-zero probability across all possible pairs.

> **Algorithm: Monte Carlo ES**
> 1. Initialize $Q(s, a) \in \mathbb{R}$ and $\pi(s) \in \mathcal{A}(s)$ arbitrarily.
> 2. Initialize $\text{Returns}(s, a) \gets \text{empty list}$.
> 3. **Loop forever (for each episode):**
>    * Choose initial pair $S_0 \in \mathcal{S}, A_0 \in \mathcal{A}(S_0)$ such that all pairs have probability $> 0$.
>    * Generate an episode from $S_0, A_0$ following $\pi$.
>    * $G \gets 0$
>    * **Loop for each step $t = T-1, T-2, \dots, 0$:**
>      * $G \gets \gamma G + R_{t+1}$
>      * **If** pair $(S_t, A_t)$ does not appear in $(S_0, A_0), \dots, (S_{t-1}, A_{t-1})$:
>        * Append $G$ to $\text{Returns}(S_t, A_t)$
>        * $Q(S_t, A_t) \gets \text{average}(\text{Returns}(S_t, A_t))$
>        * $\pi(S_t) \gets \arg\max_a Q(S_t, a)$

---

### On-Policy Control Without Exploring Starts

The Exploring Starts assumption is often unrealistic in real-world scenarios where environments dictate start states (e.g., chess or physical robotics).

To avoid Exploring Starts while maintaining exploration, **on-policy methods** evaluate or improve the same policy used to make decisions, ensuring the policy remains **stochastic** ($\varepsilon$-soft).

#### $\varepsilon$-Greedy Policies
In an $\varepsilon$-soft policy, $\pi(a \mid s) \ge \frac{\varepsilon}{|\mathcal{A}(s)|}$ for all states and actions. 

An $\varepsilon$-greedy policy assigns:
* Probability $1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|}$ to the greedy action $\arg\max_a Q(s, a)$.
* Probability $\frac{\varepsilon}{|\mathcal{A}(s)|}$ to each non-greedy action.

By the Policy Improvement Theorem, greedifying an $\varepsilon$-soft policy relative to its action-value function yields a policy guaranteed to be equal to or better than the previous one within the class of $\varepsilon$-soft policies.

---

### Off-Policy Prediction via Importance Sampling

**Off-policy methods** separate the control policy from the learning policy:
* **Target Policy ($\pi$):** The policy being learned and evaluated (typically greedy and deterministic).
* **Behavior Policy ($b$ or $\mu$):** The stochastic policy generating environment samples/actions to maintain exploration.

#### Coverage Assumption
Off-policy learning requires that any action taken by the target policy $\pi$ must also be taken at least occasionally by the behavior policy $b$:

$$\pi(a \mid s) > 0 \implies b(a \mid s) > 0 \quad \forall s \in \mathcal{S}, a \in \mathcal{A}(s)$$

#### Importance Sampling Ratio
Because returns $G_t$ are generated under sampling distribution $b$, they must be reweighted to estimate expected values under target distribution $\pi$. 

The **importance-sampling ratio** $\rho_{t:T-1}$ measures the relative probability of a trajectory occurring under $\pi$ versus $b$:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

The reweighted return product yields an unbiased value for $\pi$:

$$\mathbb{E}_b [\rho_{t:T-1} G_t \mid S_t = s] = v_\pi(s)$$

#### Ordinary vs. Weighted Importance Sampling

Given time steps $\mathcal{T}(s)$ where state $s$ was visited:

| Feature | Ordinary Importance Sampling | Weighted Importance Sampling |
| :--- | :--- | :--- |
| **Bias** | Unbiased ($\mathbb{E}[V(s)] = v_\pi(s)$) | Biased (asymptotically unbiased) |
| **Variance** | Unbounded / High (can be infinite) | Bounded / Low (preferred in practice) |

---

### Quiz

#### Question 1
Consider an episode with state trace $S_0, S_1, S_0, S_2$ and observed return $G_0 = 10$. In a First-Visit MC evaluation of state $S_0$, which visit index triggers an update to $\text{Returns}(S_0)$?

***Answer:*** Only the **first visit** at time step $t = 0$. The second visit at time step $t = 2$ is ignored by First-Visit MC.

---

#### Question 2
Why must Monte Carlo control methods estimate action values $q_\pi(s, a)$ rather than state values $v_\pi(s)$ when operating in a model-free setting?

***Answer:*** Without an explicit transition model $p(s', r \mid s, a)$, knowing $v_\pi(s)$ does not allow an agent to look ahead one step to pick the optimal action. Estimating $q_\pi(s, a)$ directly allows policy improvement via $\arg\max_a Q(s, a)$ without knowing environment dynamics.

---

#### Question 3
Which approach to maintaining exploration is more realistic for training a Blackjack playing agent: Exploring Starts or $\varepsilon$-greedy action selection?

***Answer:*** **$\varepsilon$-greedy action selection**. Exploring Starts requires initializing episodes at arbitrary, non-standard state-action pairs (e.g., starting a hand with specific total sums and dealer up-cards), which cannot be controlled in real physical card dealing.

---

#### Question 4
Categorize the following scenario: An autonomous vehicle drives using a safe, highly stochastic human driver policy while recording logs. Later, an algorithm evaluates the expected long-term safety return of a distinct, fully automated deterministic policy using those recorded logs.

***Answer:*** **Off-policy evaluation**. The target policy (fully automated) differs from the behavior policy (human driver) that generated the experience data.

---

#### Question 5
Suppose target policy $\pi$ is deterministic and selects action $A_1$ in state $S_1$ ($\pi(A_1 \mid S_1) = 1.0$). Behavior policy $b$ is uniform random over 4 actions ($b(a \mid S_1) = 0.25$). Calculate the importance sampling ratio $\rho_{0:0}$ for a single-step trajectory where action $A_1$ was selected.

***Answer:*** $$\rho_{0:0} = \frac{\pi(A_0 \mid S_0)}{b(A_0 \mid S_0)} = \frac{1.0}{0.25} = 4.0$$

---

### Sources

* **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 5: Monte Carlo Methods.
* **Morales, M.** (2020). *Grokking Deep Reinforcement Learning*. Manning Publications.