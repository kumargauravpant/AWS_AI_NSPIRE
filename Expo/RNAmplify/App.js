//import { StatusBar } from 'expo-status-bar';
import React from "react"
import { StyleSheet, Text, View } from "react-native"
import Amplify from "aws-amplify"
import config from "./src/aws-exports"

// New ----
import { withAuthenticator } from "aws-amplify-react-native"

Amplify.configure(config)

class App extends React.Component {
	render() {
		return (
			<View style={styles.container}>
				<Text>React Native + Amplify = 💛</Text>
			</View>
		)
	}
}

const styles = StyleSheet.create({
	container: {
		flex: 1,
		backgroundColor: "#fff",
		alignItems: "center",
		justifyContent: "center"
	}
})

// New ----
export default withAuthenticator(App, true)