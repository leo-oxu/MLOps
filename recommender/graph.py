from langgraph.graph import StateGraph, END
from agents import *

MAX_ROUNDS = 5

def parse_ready_flag(text: str) -> bool:
    """Detect a strict [READY] marker at end of text."""
    return bool(re.search(r"\[READY\]\s*$", text))
# LangGraph Definition

def build_graph():
    pref = preference_analyst_agent()
    rec_a = recommender_agent("A")
    rec_b = recommender_agent("B")
    ref_a = refiner_agent("A")
    ref_b = refiner_agent("B")
    crit_a = critic_agent("A")
    crit_b = critic_agent("B")
    mng_a = manager_agent("A")
    mng_b = manager_agent("B")
    eval_node = evaluator_agent()
    fb_node = feedback_agent()
    fact_checker = fact_checker_agent()
    
    graph = StateGraph()

    graph.add_node("analyze_preferences", lambda state:{
        **state,
        "query": pref.invoke({"user_input": state["user_input"]}),
        "round": state.get("round", 1),
    })

    graph.add_node("recommender_a", lambda state: {
        **state,
        "rag_context_a": retriever.get_relevant_documents(state["query"]),
        "web_a": tavily.run(state["query"]),
        "proposal_a":rec_a.invoke({
            "query":state["query"],
            "context": "\n".join([doc.page_content for doc in state["rag_context_a"]]),
            "web": state["web_a"]
        }),
        "ref_prop_a": state["proposal_a"],
        "critique_b": "",
        "fc_critique_b": "",
    })

    graph.add_node("recommender_b", lambda state: {
        **state,
        "rag_context_b": retriever.get_relevant_documents(state["query"]),
        "web_b": tavily.run(state["query"]),
        "proposal_b": rec_b.invoke({
            "query": state["query"],
            "context": state["rag_context_b"],
            "web": state["web_b"]
        }),
        "ref_prop_b": state["proposal_b"],
        "critique_a": "",
        "fc_critique_a": "",
    })
    
    graph.add_node("refiner_a", lambda state: {
        **state,
        "web_ref_a": tavily.run(f"Gather info for enhanced proposal based on opponent critique: \n{state["critique_b"]} \n and fact checks on opponent critique: \n {state["fc_critique_b"]}"),
        "ref_prop_a": ref_a.invoke({
            "proposal": state["ref_prop_a"],
            "critique": state["critique_b"],
            "fc_critique": state['fc_critique_b']
        }),
        "round": state["round"] + 1
    })

    graph.add_node("refiner_b", lambda state: {
        **state,
        "web_ref_b": tavily.run(f"Gather info for enhanced proposal based on opponent critique: \n{state["critique_a"]} \n and fact checks on opponent critique: \n {state["fc_critique_a"]}"),
        "ref_prop_b": ref_a.invoke({
            "proposal": state["ref_prop_b"],
            "critique": state["critique_a"],
            "fc_critique": state["fc_critique_a"]
        })
    })

    graph.add_node("fact_checker_b", lambda state:{
        **state,
        "fc_proposal_a": fact_checker.invoke({
            "claim": state["proposal_a"],
            "web": tavily.run(f"verify {state['proposal_a']}")
        })
    })

    graph.add_node("fact_check_a", lambda state:{
        **state,
        "fc_proposal_b": fact_checker.invoke({
            "claim": state["proposal_b"],
            "web": tavily.run(f"verify {state['proposal_b']}")
        })
    })

    graph.add_node("critique_a", lambda state:{
        **state,
        "critique_a": crit_a.invoke({
            "own_proposal": state["proposal_a"],
            "opponent_proposal": state["proposal_b"],
            "fc_opponent_proposal": state["fc_proposal_b"],
            "web": tavily.run(f"Negative aspects of {state["opponent_proposal"]}") 
        })
    })

    graph.add_node("critique_b", lambda state: {
        **state,
        "critique_b": crit_b.invoke({
            "own_proposal": state["proposal_b"],
            "opponent_proposal": state["proposal_a"],
            "fc_opponent_proposal": state["fc_opponent_a"],
            "web": tavily.run(f"Negative aspect of {state["opponent_proposal"]}") 
        })
    })

    graph.add_node("manager_a", lambda state:{
        **state,
        "decision_a": mng_a.invoke({
            "own_proposal": state["ref_prop_a"],
            "opponent_proposal": state["ref_prop_b"],
            "own_critique": state["critique_a"],
            "opponent_critique":state["crtique_b"]
        }),
        "ready_a": parse_ready_flag(state["decision_a"])
    })

    graph.add_node("manager_b", lambda state:{
        **state,
        "decision_b": mng_b.invoke({
            "own_proposal": state["ref_prop_b"],
            "opponent_proposal": state["ref_prop_a"],
            "own_critique": state["critique_b"],
            "opponent_critique": state["critique_a"]
        }),
        "ready_b": parse_ready_flag(state["decision_b"])
    })

    graph.add_node("evaluate", lambda state: {
        **state,
        "recommendation": eval_node.invoke({
            "proposal_a": state["proposal_a"],
            "proposal_b": state["proposal_b"],
            "critique_a": state["critique_a"],
            "critique_b": state["critique_b"]
        })
    })

    def feedback_handler(state):
        user_reply = fb_node.invoke({"recommendation": state["recommendation"]})
        if "stop" in user_reply.lower():
            return{**state,"continue": False}
        return {**state, "continue": True, "user_input": user_reply}
    
    graph.add_node("user_feedback", feedback_handler)

    graph.set_entry_point("analyze_preferences")
    graph.add_edge("analyze_preferences","recommender_a")
    graph.add_edge("analyze_pereferences","recommneder_b")
    graph.add_edge("recommender_a","refiner_a")
    graph.add_edge("recommender_b", "refiner_b")
    graph.add_edge("refiner_a", "fact_checker_a")
    graph.add_edge("refiner_b","fact_checker_b")
    graph.add_edge("fact_checker_a","critique_a")
    graph.add_edge("fact_checker_b","critique_b")
    graph.add_edge("critique_a", "manager_a")
    graph.add_edge("critique_b", "manager_b")
    graph.add_conditional_edges(
        "manager_a",
        condition=lambda state: state["ready_a"] and state["ready_b"] or state["round"] >= 5,
        path_map= {
            True:"refiner_a",
            False: END
        }
    )
    
    graph.add_conditional_edges(
        "manager_b",
        condition=lambda state: state["ready_a"] and state["ready_b"] or state["round"] >= 5,
        path_map= {
            True:"refiner_b",
            False: END
        }
    )
    graph.add_edge("evaluate","user_feedback")

    graph.add_conditional_edges(
        "user_feedback",
        condition=lambda state: state["continue"],
        path_map={
            True:"recommender",
            False: "evaluator"
        }
    )
    

    return graph.compile()

if __name__ == "__main__":
    travel_graph = build_graph()
    state = {
        "user_input": "I want to travel in summer to a place with beaches, culture, warm weather, and moderate budget"
    }
    result = travel_graph.invoke(state)
    print("\n\n🛫 Final Recommendation:")
    print(result.get("recommendation", "No result"))
    print("\n✍️ User Feedback:")
    print(result.get("user_feedback", "No feedback"))